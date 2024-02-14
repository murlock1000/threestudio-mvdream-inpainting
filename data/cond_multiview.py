import copy
import json
import math
import os
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import tqdm
from threestudio import register
import threestudio
from threestudio.data.multiview import MultiviewDataModule, MultiviewDataset, MultiviewIterableDataset, MultiviewsDataModuleConfig, convert_proj
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *
from torch.utils.data import DataLoader, Dataset, IterableDataset

c2w_quadruplets = []

def write_c2w():
    global c2w_quadruplets
    torch.save(c2w_quadruplets, 'c2w_quadruplets.pt')

@dataclass
class MVDreamMultiviewsDataModuleConfig(MultiviewsDataModuleConfig):
    # Dataset parameters
    n_view: int = 4
    crop_to: int = 1024
    input_size: int = 256
    novel_frame_count: int = 1
    train_split: str = "train"

    enableLateMV: bool = True
    startMVAt: int = 500
    stopMVAt: int = 900
    enableProbabilisticMV: bool = False
    MVProbability: float = .5
    use_fib_generator: bool = True
    max_fib_poses: int = 1000

    # Random camera parameters
    relative_radius: bool = True
    zoom_range: Tuple[float, float] = (1.0, 1.0)
    rays_d_normalize: bool = True
    height: Any = 64
    width: Any = 64
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    n_val_views: int = 1
    n_test_views: int = 120
    elevation_range: Tuple[float, float] = (-10, 90)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy

def normalize_camera(camera_matrix):
    ''' normalize the camera location onto a unit-sphere'''
    if isinstance(camera_matrix, np.ndarray):
        camera_matrix = camera_matrix.reshape(-1,4,4)
        translation = camera_matrix[:,:3,3]
        translation = translation / (np.linalg.norm(translation, axis=1, keepdims=True) + 1e-8)
        camera_matrix[:,:3,3] = translation
    else:
        camera_matrix = camera_matrix.reshape(-1,4,4)
        translation = camera_matrix[:,:3,3]
        translation = translation / (torch.norm(translation, dim=1, keepdim=True) + 1e-8)
        camera_matrix[:,:3,3] = translation
    return camera_matrix.reshape(-1,16)

def crop_center(img, w = 1024, h = 1024):
        # Get the current size of the image
        center = img.shape

        # Calculate the coordinates for cropping
        x = center[1]/2 - w/2
        y = center[0]/2 - h/2

        return img[int(y):int(y+h), int(x):int(x+w)]

class RandomMultiviewCameraIterableDataset(RandomCameraIterableDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zoom_range = self.cfg.zoom_range

    def collate(self, batch) -> Dict[str, Any]:
        assert (
            self.batch_size % self.cfg.n_view == 0
        ), f"batch_size ({self.batch_size}) must be dividable by n_view ({self.cfg.n_view})!"
        real_batch_size = self.batch_size // self.cfg.n_view

        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(real_batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(real_batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        # ensures sampled azimuth angles in a batch cover the whole range
        azimuth_deg = (
            torch.rand(real_batch_size).reshape(-1, 1)
            + torch.arange(self.cfg.n_view).reshape(1, -1)
        ).reshape(-1) / self.cfg.n_view * (
            self.azimuth_range[1] - self.azimuth_range[0]
        ) + self.azimuth_range[
            0
        ]
        azimuth = azimuth_deg * math.pi / 180

        ######## Different from original ########
        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        if self.cfg.relative_radius:
            scale = 1 / torch.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

        # zoom in by decreasing fov after camera distance is fixed
        zoom: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.zoom_range[1] - self.zoom_range[0])
            + self.zoom_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy * zoom
        fovy_deg = fovy_deg * zoom
        ###########################################

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(real_batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.center_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.up_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        up = up + up_perturb

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(real_batch_size, 3).repeat_interleave(
                    self.cfg.n_view, dim=0
                )
                * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(real_batch_size) * math.pi - 2 * math.pi
            ).repeat_interleave(
                self.cfg.n_view, dim=0
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(real_batch_size) * math.pi / 3 + math.pi / 6
            ).repeat_interleave(
                self.cfg.n_view, dim=0
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, normalize=self.cfg.rays_d_normalize
        )

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        #global c2w_quadruplets
        #c2w_quadruplets.append(c2w)

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "fovy": fovy,
        }

class MVDreamMultiviewDataset(Dataset):
   
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: MVDreamMultiviewsDataModuleConfig = cfg

        camera_dict = json.load(
            open(os.path.join(self.cfg.dataroot, f"transforms_{split}.json"), "r")
        )
        self.evaluator = Evaluator(split)
        frames = camera_dict["frames"]
        camera_angle_x = camera_dict["camera_angle_x"]
        camera_angle_y = camera_dict["camera_angle_y"]
        fl_x = camera_dict["fl_x"]
        fl_y = camera_dict["fl_y"]
        k1 = camera_dict["k1"]
        k2 = camera_dict["k2"]
        p1 = camera_dict["p1"]
        p2 = camera_dict["p2"]
        cx = camera_dict["cx"]
        cy = camera_dict["cy"]
        w = camera_dict["w"] # 1440
        h = camera_dict["h"] # 1080
        aabb_scale = camera_dict["aabb_scale"]

        wScale =  self.cfg.crop_to // self.cfg.input_size
        hScale =  self.cfg.crop_to // self.cfg.input_size

        frames_proj = []
        frames_c2w = []
        frames_position = []
        frames_direction = []
        frames_img = []

        self.frame_w = self.cfg.input_size
        self.frame_h = self.cfg.input_size
        threestudio.info("Loading frames...")
        self.n_frames = len(frames)

        c2w_list = []
        for frame in frames:
            extrinsic: Float[Tensor, "4 4"] = torch.as_tensor(
                frame["transform_matrix"], dtype=torch.float32
            )
            # Normalize camera to unit sphere
            c2w = normalize_camera(extrinsic).reshape(4,4)
            c2w_list.append(c2w)
        c2w_list = torch.stack(c2w_list, dim=0)


        for idx, frame in enumerate(frames):
            intrinsic: Float[Tensor, "4 4"] = torch.eye(4)
            intrinsic[0, 0] = fl_x / wScale # Cropping does not change focal point, scaling does.
            intrinsic[1, 1] = fl_y / hScale 
            intrinsic[0, 2] = (cx - (w-self.cfg.crop_to)/2) / wScale # Cropping reduces cx,cy by pixels cropped in top and left.
            intrinsic[1, 2] = (cy - (h-self.cfg.crop_to)/2) / hScale

            frame_path = os.path.join(self.cfg.dataroot, frame["file_path"]+".png")
            img = cv2.imread(frame_path)
            
            img = crop_center(img, self.cfg.crop_to, self.cfg.crop_to)

            img = cv2.resize(img, (self.frame_w, self.frame_h))

            transparency_mask = img[:, :, -1].copy()
            transparency_mask: Float[Tensor, "H W 1"] = torch.FloatTensor(~(transparency_mask == 0)).unsqueeze(dim=-1) # Boolean transparency mask with 0 - transparent 1 - opaque (rgb)

            img = img[:, :, ::-1].copy()
            img: Float[Tensor, "H W 3"] = torch.FloatTensor(img) / 255

            white_bck = torch.ones_like(img)
            img = img * transparency_mask + white_bck * (1. - transparency_mask) # Set background to white for validation

            frames_img.append(img)

            direction: Float[Tensor, "H W 3"] = get_ray_directions(
                self.frame_h,
                self.frame_w,
                (intrinsic[0, 0], intrinsic[1, 1]),
                (intrinsic[0, 2], intrinsic[1, 2]),
                use_pixel_centers=False,
            )

            c2w = c2w_list[idx]
            camera_position: Float[Tensor, "3"] = c2w[:3, 3:].reshape(-1)

            near = 0.1
            far = 1000.0
            proj = convert_proj(intrinsic, self.frame_h, self.frame_w, near, far)
            proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
            frames_proj.append(proj)
            frames_c2w.append(c2w)
            frames_position.append(camera_position)
            frames_direction.append(direction)
            
        ## Generate random views in cardinal directions for animations
        self.randomDataset = RandomCameraDataset(cfg, split)

        threestudio.info("Loaded frames.")

        self.frames_proj: Float[Tensor, "B 4 4"] = torch.stack(frames_proj, dim=0)
        self.frames_c2w: Float[Tensor, "B 4 4"] = torch.stack(frames_c2w, dim=0)
        self.frames_position: Float[Tensor, "B 3"] = torch.stack(frames_position, dim=0)
        self.frames_direction: Float[Tensor, "B H W 3"] = torch.stack(
            frames_direction, dim=0
        )
        self.frames_img: Float[Tensor, "B H W 3"] = torch.stack(frames_img, dim=0)

        self.rays_o, self.rays_d = get_rays(
            self.frames_direction,
            self.frames_c2w,
            keepdim=True,
            normalize=self.cfg.rays_d_normalize,
        )
        self.mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(
            self.frames_c2w, self.frames_proj
        )
        self.light_positions: Float[Tensor, "B 3"] = torch.zeros_like(
            self.frames_position
        )

        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )

        self.batch_size: int = self.batch_sizes[0]


    def __len__(self):
        return self.frames_proj.shape[0]+len(self.randomDataset)

    def __getitem__(self, index):
        if index >= self.randomDataset.n_views:
            index = index - self.randomDataset.n_views
            return {
                "index": index+self.randomDataset.n_views,
                "rays_o": self.rays_o[index],
                "rays_d": self.rays_d[index],
                "mvp_mtx": self.mvp_mtx[index],
                "c2w": self.frames_c2w[index],
                "camera_positions": self.frames_position[index],
                "light_positions": self.light_positions[index],
                "gt_rgb": self.frames_img[index],
            }
        
        novel_view = self.randomDataset.__getitem__(index)
        return {
            "index": index,
            "rays_o": novel_view["rays_o"],
            "rays_d": novel_view["rays_d"],
            "mvp_mtx": novel_view["mvp_mtx"],
            "c2w": novel_view["c2w"],
            "camera_positions": novel_view["camera_positions"],
            "light_positions": novel_view["light_positions"],
            "gt_rgb": torch.ones_like(self.frames_img[0]),
        }

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.frame_h, "width": self.frame_w})
        batch.update({"stop_save_at": self.randomDataset.n_views})
        batch.update({"evaluator": self.evaluator})
        return batch

class NovelFrames():
    # Using spherical cameras seems to break the model at the 202 iteration mark where it freezes.
    def fibonacci_sphere(samples=1000):
        points = []
        phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points.append((x, y, z))

        return points

    def create_camera_to_world_matrix(elevation, azimuth):
        elevation = np.radians(elevation)
        azimuth = np.radians(azimuth)
        # Convert elevation and azimuth angles to Cartesian coordinates on a unit sphere
        x = np.cos(elevation) * np.sin(azimuth)
        z = np.sin(elevation)
        y = np.cos(elevation) * np.cos(azimuth)

        # Calculate camera position, target, and up vectors
        camera_pos = np.array([x, y, z])
        target = np.array([0, 0, 0])
        up = np.array([0, 1, 0])

        # Construct view matrix
        forward = target - camera_pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        new_up = np.cross(right, forward)
        new_up /= np.linalg.norm(new_up)
        cam2world = np.eye(4)
        cam2world[:3, :3] = np.array([right, new_up, -forward]).T
        cam2world[:3, 3] = camera_pos

        if np.isnan(cam2world).any():
            return None
        return cam2world
    
    def fibonacci_northern_hemisphere(samples=1000):
        samples = samples * 2
        points = []
        phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

        for i in range(samples//2):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y**2)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            z, y = y, z
            points.append((x, y, z))

        return points

    def create_camera_to_world_matrix_fib(position):
        position /= np.linalg.norm(position)

        # Calculate camera position, target, and up vectors
        camera_pos = position
        target = np.array([0, 0, 0])
        up = np.array([0, 1, 0])

        # Construct view matrix
        forward = target - camera_pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        new_up = np.cross(right, forward)
        new_up /= np.linalg.norm(new_up)
        cam2world = np.eye(4)
        cam2world[:3, :3] = np.array([right, new_up, -forward]).T
        cam2world[:3, 3] = camera_pos
        return torch.as_tensor(
                cam2world, dtype=torch.float32
        )

    def __init__(self, cfg, camera_dict, num_poses=100):
        self.cfg: MVDreamMultiviewsDataModuleConfig = cfg

        fl_x = camera_dict["fl_x"]
        fl_y = camera_dict["fl_y"]
        cx = camera_dict["cx"]
        cy = camera_dict["cy"]
        w = camera_dict["w"]
        h = camera_dict["h"]

        #self.cfg.input_size = 32

        wScale = self.cfg.crop_to // self.cfg.input_size
        hScale = self.cfg.crop_to // self.cfg.input_size

        frames_proj = []
        frames_c2w = []
        frames_position = []
        frames_direction = []
        frames_img = []

        self.frame_w = self.cfg.input_size 
        self.frame_h = self.cfg.input_size

        threestudio.info("Generating novel frames...")
        self.n_frames = num_poses

        #positions = NovelFrames.fibonacci_northern_hemisphere(self.n_frames)
        #positions = create_camera_to_world_matrix
        angle_gap = 2
        azimuth_start = 0
        azimuth_span = 360
        elevation_start = 0
        elevation_span = 30

        azimuths = [azimuth for azimuth in np.arange(azimuth_start, azimuth_span+azimuth_start+angle_gap, angle_gap)]
        elevations =  [elevation for elevation in np.arange(elevation_start, elevation_span+elevation_start+angle_gap, angle_gap)]
        autogen_sampled_poses = [NovelFrames.create_camera_to_world_matrix(elevation, azimuth) for azimuth in azimuths for elevation in elevations]
        autogen_sampled_poses = [torch.as_tensor(
                x, dtype=torch.float32
        ) for x in autogen_sampled_poses if x is not None]

        self.n_frames = len(autogen_sampled_poses)
        #autogen_sampled_poses = [NovelFrames.create_camera_to_world_matrix_fib(pos) for pos in positions]

        c2w_list = torch.stack(autogen_sampled_poses, dim=0)

        for idx in range(self.n_frames):
            intrinsic: Float[Tensor, "4 4"] = torch.eye(4)
            intrinsic[0, 0] = fl_x / wScale # Cropping does not change focal point, scaling does.
            intrinsic[1, 1] = fl_y / hScale 
            intrinsic[0, 2] = (cx - (w-self.cfg.crop_to)/2) / wScale # Cropping reduces cx,cy by pixels cropped in top and left.
            intrinsic[1, 2] = (cy - (h-self.cfg.crop_to)/2) / hScale

            direction: Float[Tensor, "H W 3"] = get_ray_directions(
                self.frame_h,
                self.frame_w,
                (intrinsic[0, 0], intrinsic[1, 1]),
                (intrinsic[0, 2], intrinsic[1, 2]),
                use_pixel_centers=False,
            )

            c2w = c2w_list[idx]
            camera_position: Float[Tensor, "3"] = c2w[:3, 3:].reshape(-1)
            near = 0.1
            far = 1000.0
            proj = convert_proj(intrinsic, self.frame_h, self.frame_w, near, far)
            proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
            frames_proj.append(proj)
            frames_c2w.append(c2w)
            frames_position.append(camera_position)
            frames_direction.append(direction)
            frames_img.append(torch.tensor([]))

        self.frames_proj: Float[Tensor, "B 4 4"] = torch.stack(frames_proj, dim=0)
        self.frames_c2w: Float[Tensor, "B 4 4"] = torch.stack(frames_c2w, dim=0)
        self.frames_position: Float[Tensor, "B 3"] = torch.stack(frames_position, dim=0)
        self.frames_direction: Float[Tensor, "B H W 3"] = torch.stack(
            frames_direction, dim=0
        )
        self.frames_img: Float[Tensor, "B H W 3"] = torch.stack(frames_img, dim=0)

        self.rays_o, self.rays_d = get_rays(
            self.frames_direction,
            self.frames_c2w,
            keepdim=True,
            normalize=self.cfg.rays_d_normalize,
        )
        self.mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(
            self.frames_c2w, self.frames_proj
        )
        self.light_positions: Float[Tensor, "B 3"] = torch.zeros_like(
            self.frames_position
        )

class MVDreamMultiviewIterableDataset(IterableDataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: MVDreamMultiviewsDataModuleConfig = cfg

        camera_dict = json.load(
            open(os.path.join(self.cfg.dataroot, f"transforms_{split}.json"), "r")
        )
        self.train_step = 0
        self.split = split
        frames = camera_dict["frames"]
        camera_angle_x = camera_dict["camera_angle_x"]
        camera_angle_y = camera_dict["camera_angle_y"]
        fl_x = camera_dict["fl_x"]
        fl_y = camera_dict["fl_y"]
        k1 = camera_dict["k1"]
        k2 = camera_dict["k2"]
        p1 = camera_dict["p1"]
        p2 = camera_dict["p2"]
        cx = camera_dict["cx"]
        cy = camera_dict["cy"]
        w = camera_dict["w"]
        h = camera_dict["h"]
        aabb_scale = camera_dict["aabb_scale"]

        #self.cfg.input_size = 32

        wScale = self.cfg.crop_to // self.cfg.input_size
        hScale = self.cfg.crop_to // self.cfg.input_size

        frames_proj = []
        frames_c2w = []
        frames_position = []
        frames_direction = []
        frames_img = []
        transparency_masks = []

        self.frame_w = self.cfg.input_size 
        self.frame_h = self.cfg.input_size
        threestudio.info("Loading frames...")
        self.n_frames = len(frames)

        c2w_list = []
        for frame in frames:
            extrinsic: Float[Tensor, "4 4"] = torch.as_tensor(
                frame["transform_matrix"], dtype=torch.float32
            )
            # Normalize camera to unit sphere
            c2w = normalize_camera(extrinsic).reshape(4,4)
            c2w_list.append(c2w)
        c2w_list = torch.stack(c2w_list, dim=0)


        for idx, frame in enumerate(frames):
            intrinsic: Float[Tensor, "4 4"] = torch.eye(4)
            intrinsic[0, 0] = fl_x / wScale # Cropping does not change focal point, scaling does.
            intrinsic[1, 1] = fl_y / hScale 
            intrinsic[0, 2] = (cx - (w-self.cfg.crop_to)/2) / wScale # Cropping reduces cx,cy by pixels cropped in top and left.
            intrinsic[1, 2] = (cy - (h-self.cfg.crop_to)/2) / hScale

            frame_path = os.path.join(self.cfg.dataroot, frame["file_path"]+".png")
            img = cv2.imread(frame_path)
            
            img = crop_center(img, self.cfg.crop_to, self.cfg.crop_to)

            img = cv2.resize(img, (self.frame_w, self.frame_h))

            transparency_mask = img[:, :, -1].copy() 
            transparency_mask: Float[Tensor, "H W 1"] = torch.FloatTensor(~(transparency_mask == 0)).unsqueeze(dim=-1) # Boolean transparency mask
            transparency_masks.append(transparency_mask) # 0 - transparent 1 - opaque (rgb)

            img = img[:, :, ::-1].copy()

            img: Float[Tensor, "H W 3"] = torch.FloatTensor(img) / 255
            frames_img.append(img)

            direction: Float[Tensor, "H W 3"] = get_ray_directions(
                self.frame_h,
                self.frame_w,
                (intrinsic[0, 0], intrinsic[1, 1]),
                (intrinsic[0, 2], intrinsic[1, 2]),
                use_pixel_centers=False,
            )

            c2w = c2w_list[idx]
            camera_position: Float[Tensor, "3"] = c2w[:3, 3:].reshape(-1)

            near = 0.1
            far = 1000.0
            proj = convert_proj(intrinsic, self.frame_h, self.frame_w, near, far)
            proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
            frames_proj.append(proj)
            frames_c2w.append(c2w)
            frames_position.append(camera_position)
            frames_direction.append(direction)
            
        threestudio.info("Loaded frames.")

        self.frames_proj: Float[Tensor, "B 4 4"] = torch.stack(frames_proj, dim=0)
        self.frames_c2w: Float[Tensor, "B 4 4"] = torch.stack(frames_c2w, dim=0)
        self.frames_position: Float[Tensor, "B 3"] = torch.stack(frames_position, dim=0)
        self.frames_direction: Float[Tensor, "B H W 3"] = torch.stack(
            frames_direction, dim=0
        )
        self.frames_img: Float[Tensor, "B H W 3"] = torch.stack(frames_img, dim=0)
        self.transparency_masks: Float[Tensor, "B H W 1"] = torch.stack(transparency_masks, dim=0)

        self.rays_o, self.rays_d = get_rays(
            self.frames_direction,
            self.frames_c2w,
            keepdim=True,
            normalize=self.cfg.rays_d_normalize,
        )
        self.mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(
            self.frames_c2w, self.frames_proj
        )
        self.light_positions: Float[Tensor, "B 3"] = torch.zeros_like(
            self.frames_position
        )

        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )

        self.batch_size: int = self.batch_sizes[0]

        if self.cfg.use_fib_generator:
            self.novel_frames = NovelFrames(self.cfg, camera_dict, self.cfg.max_fib_poses)
        else:
            novel_cfg = copy.copy(self.cfg)
            novel_cfg.width = [self.cfg.input_size, self.cfg.input_size]
            novel_cfg.height = [self.cfg.input_size, self.cfg.input_size]
            self.novel_generator = RandomMultiviewCameraIterableDataset(novel_cfg)

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        assert (
            self.batch_size % self.cfg.n_view == 0
        ), f"batch_size ({self.batch_size}) must be dividable by n_view ({self.cfg.n_view})!"
        real_batch_size = self.batch_size // self.cfg.n_view

        self.train_step += 1
        
        novel_frame_count = self.cfg.novel_frame_count
        if self.cfg.enableLateMV:
            if self.train_step < self.cfg.startMVAt:
                novel_frame_count = 0

            if self.train_step >= self.cfg.stopMVAt:
                novel_frame_count = 0
        
        if self.cfg.enableProbabilisticMV:
            if self.cfg.MVProbability > random.random():
                novel_frame_count = 0
            
                
        # Use 1 GT images from train set and 1 novel view in batch for guidance: [novel, gt, gt, gt]
        if novel_frame_count < self.cfg.n_view:
            train_indexes = torch.randperm(self.n_frames)[:(self.cfg.n_view-novel_frame_count)]
            index = torch.cat((torch.tensor([-1]), train_indexes))
        else:
            train_indexes = []
            index = 0
    
        #if self.split == "train":
        #    train = torch.randint(0, 16, (3,)) 
        #    novel = torch.randint(16, self.n_frames, (1,))
        #    index = torch.cat((novel, train))
        #else:
        #    index = torch.randint(0, self.n_frames, (4,))

        if novel_frame_count != 0:
            if not self.cfg.use_fib_generator:
                self.novel_frames = self.novel_generator.collate(None)
                self.novel_frames["n_frames"] = 4
                novel_idx = torch.randint(self.novel_frames["n_frames"], (novel_frame_count,))
            else:
                novel_idx = torch.randint(self.novel_frames.n_frames, (novel_frame_count,))
        else:
            novel_idx = []

        if novel_frame_count == 0:
            return {
                "index": index,
                "rays_o": self.rays_o[train_indexes],
                "rays_d": self.rays_d[train_indexes],
                "mvp_mtx": self.mvp_mtx[train_indexes],
                "c2w": self.frames_c2w[train_indexes],
                "camera_positions": self.frames_position[train_indexes],
                "light_positions": self.light_positions[train_indexes],
                "transparency_masks": self.transparency_masks[train_indexes],
                "gt_rgb": self.frames_img[train_indexes],
                "height": self.frame_h,
                "width": self.frame_w,
                "elevation": None,
                "azimuth": None,
                "camera_distances": None,
                "fovy": None, # Not used by model for camera conditioning
                "novel_frame_count": novel_frame_count,
            }
        elif self.cfg.use_fib_generator:
            return {
                "index": index,
                "rays_o": torch.cat((self.novel_frames.rays_o[novel_idx], self.rays_o[train_indexes])),
                "rays_d": torch.cat((self.novel_frames.rays_d[novel_idx], self.rays_d[train_indexes])),
                "mvp_mtx": torch.cat((self.novel_frames.mvp_mtx[novel_idx], self.mvp_mtx[train_indexes])),
                "c2w": torch.cat((self.novel_frames.frames_c2w[novel_idx], self.frames_c2w[train_indexes])),
                "camera_positions": torch.cat((self.novel_frames.frames_position[novel_idx], self.frames_position[train_indexes])),
                "light_positions": torch.cat((self.novel_frames.light_positions[novel_idx], self.light_positions[train_indexes])),
                "transparency_masks": self.transparency_masks[train_indexes],
                "gt_rgb": self.frames_img[train_indexes],
                "height": self.frame_h,
                "width": self.frame_w,
                "elevation": None,
                "azimuth": None,
                "camera_distances": None,
                "fovy": None, # Not used by model for camera conditioning
                "novel_frame_count": novel_frame_count,
            }
        else:
            return {
                "index": index,
                "rays_o": torch.cat((self.novel_frames["rays_o"][novel_idx], self.rays_o[train_indexes])),
                "rays_d": torch.cat((self.novel_frames["rays_d"][novel_idx], self.rays_d[train_indexes])),
                "mvp_mtx": torch.cat((self.novel_frames["mvp_mtx"][novel_idx], self.mvp_mtx[train_indexes])),
                "c2w": torch.cat((self.novel_frames["c2w"][novel_idx], self.frames_c2w[train_indexes])),
                "camera_positions": torch.cat((self.novel_frames["camera_positions"][novel_idx], self.frames_position[train_indexes])),
                "light_positions": torch.cat((self.novel_frames["light_positions"][novel_idx], self.light_positions[train_indexes])),
                "transparency_masks": self.transparency_masks[train_indexes],
                "gt_rgb": self.frames_img[train_indexes],
                "height": self.frame_h,
                "width": self.frame_w,
                "elevation": None,
                "azimuth": None,
                "camera_distances": None,
                "fovy": None, # Not used by model for camera conditioning
                "novel_frame_count": novel_frame_count,
            }


@register("mvdream-multiview-camera-datamodule")
class MVDreamMultiviewCameraDataModule(pl.LightningDataModule):
    cfg: MVDreamMultiviewsDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MVDreamMultiviewsDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = MVDreamMultiviewIterableDataset(self.cfg, self.cfg.train_split)
        if stage in [None, "fit", "validate"]:
            #self.val_dataset = RandomCameraDataset(self.cfg, "val")
            self.val_dataset = MVDreamMultiviewDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            #self.test_dataset = RandomCameraDataset(self.cfg, "test")
            self.test_dataset = MVDreamMultiviewDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
    




#################################################################################
    
import argparse
import glob
import os
from collections import namedtuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from tqdm import tqdm

import threestudio


from threestudio.utils.perceptual.perceptual import PerceptualLoss

transform = transforms.Compose([transforms.ToTensor()])

# MIT Licence

# Methods to predict the SSIM, taken from
# https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, mask=None, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if not (mask is None):
        b = mask.size(0)
        ssim_map = ssim_map.mean(dim=1, keepdim=True) * mask
        ssim_map = ssim_map.view(b, -1).sum(dim=1) / mask.view(b, -1).sum(dim=1).clamp(
            min=1
        )
        return ssim_map

    import pdb

    pdb.set_trace

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, mask=None):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(
            img1,
            img2,
            window,
            self.window_size,
            channel,
            mask,
            self.size_average,
        )


def ssim(img1, img2, window_size=11, mask=None, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, mask, size_average)



def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1)).view(
        in_feat.size()[0], 1, in_feat.size()[2], in_feat.size()[3]
    )
    return in_feat / (norm_factor.expand_as(in_feat) + eps)


def cos_sim(in0, in1):
    in0_norm = normalize_tensor(in0)
    in1_norm = normalize_tensor(in1)
    N = in0.size()[0]
    X = in0.size()[2]
    Y = in0.size()[3]

    return torch.mean(
        torch.mean(torch.sum(in0_norm * in1_norm, dim=1).view(N, 1, X, Y), dim=2).view(
            N, 1, 1, Y
        ),
        dim=3,
    ).view(N)

# The SSIM metric
def ssim_metric(img1, img2, mask=None):
    return ssim(img1, img2, mask=mask, size_average=False)


# The PSNR metric
def psnr(img1, img2, mask=None, reshape=False):
    b = img1.size(0)
    if not (mask is None):
        b = img1.size(0)
        mse_err = (img1 - img2).pow(2) * mask
        if reshape:
            mse_err = mse_err.reshape(b, -1).sum(dim=1) / (
                3 * mask.reshape(b, -1).sum(dim=1).clamp(min=1)
            )
        else:
            mse_err = mse_err.view(b, -1).sum(dim=1) / (
                3 * mask.view(b, -1).sum(dim=1).clamp(min=1)
            )
    else:
        if reshape:
            mse_err = (img1 - img2).pow(2).reshape(b, -1).mean(dim=1)
        else:
            mse_err = (img1 - img2).pow(2).view(b, -1).mean(dim=1)

    psnr = 10 * (1 / mse_err).log10()
    return psnr


# The LPIPS metric
def perceptual_sim(img1, img2, perceptual_loss):
    dist = perceptual_loss(
        img1.permute(0, 3, 1, 2).contiguous(),
        img2.permute(0, 3, 1, 2).contiguous(),
    ).sum()

    return dist


def load_img(img_name, size=None):
    print("aa")
    try:
        img = Image.open(img_name).convert('RGB')

        if type(size) == int:
            img = img.resize((size, size))
        elif size is not None:
            img = img.resize((size[1], size[0]))

        img = transform(img).cuda()
        img = img.unsqueeze(0)
    except Exception as e:
        print("Failed at loading %s " % img_name)
        print(e)
        img = torch.zeros(1, 3, 256, 256).cuda()
        raise
    return img

@threestudio.register("mvdream-evaluator")
class Evaluator():
    perceptual_loss = PerceptualLoss().eval().to("cuda")

    def __init__(self, split):
        self.split = split
        self.batches = []
        self.values_lpips = {}
        self.values_ssim = {}
        self.values_psnr = {}

        self.avg_lpips = []
        self.avg_ssim = []
        self.avg_psnr = []

        self.last_t = 0

    def compute_perceptual_similarity(self, pred_img, tgt_img, batch):
        lpips_val = perceptual_sim(pred_img, tgt_img, Evaluator.perceptual_loss).item()
        self.values_lpips[batch].append(lpips_val)
        return lpips_val

    def compute_ssim(self, pred_img, tgt_img, batch):
        ssim_val = ssim_metric(pred_img, tgt_img).item()
        self.values_ssim[batch].append(ssim_val)
        return ssim_val
    
    def compute_psnr(self, pred_img, tgt_img, batch):
        psnr_val = psnr(pred_img, tgt_img).item()
        self.values_psnr[batch].append(psnr_val)
        return psnr_val
    
    def simple_compute(self, pred_img, tgt_img, batch):
        if batch not in self.batches:
            self.batches.append(batch)
            self.values_lpips[batch] = []
            self.values_psnr[batch] = []
            self.values_ssim[batch] = []

        #pred_img = pred_img.permute(0, 3, 1, 2)
        #tgt_img = tgt_img.permute(0, 3, 1, 2)
        #1, 3, 256, 256
       #     out["comp_rgb"].size()
       # torch.Size([1, 512, 512, 3])
       # batch["gt_rgb"].size()
       # torch.Size([1, 128, 128, 3])
        lpips_val = self.compute_perceptual_similarity(pred_img, tgt_img, batch)
        ssim_val = self.compute_ssim(pred_img, tgt_img, batch)
        psnr_val = self.compute_psnr(pred_img, tgt_img, batch)

    def calc_average(self, iter, path):
        lpips_val = 0
        psnr_val = 0
        ssim_val = 0

        for batch in self.batches:
            lpips_val += self.values_lpips[batch][self.last_t]
            psnr_val += self.values_psnr[batch][self.last_t]
            ssim_val += self.values_ssim[batch][self.last_t]

        lpips_avg = lpips_val/len(self.batches)
        self.avg_lpips.append(lpips_avg)

        psnr_avg = psnr_val/len(self.batches)
        self.avg_psnr.append(psnr_avg)

        ssim_avg = ssim_val/len(self.batches)
        self.avg_ssim.append(ssim_avg)

        print(f"{iter}: Avg LPIPS: {self.avg_lpips[self.last_t]} - Avg PSNR: {self.avg_psnr[self.last_t]} - Avg SSIM: {self.avg_ssim[self.last_t]} \n")
        #'outputs\\mvdream-sd21-rescale0.5\\Toy_T-rex@20240209-143145'
        self.write_to_file(path, iter)
        
        self.last_t = self.last_t + 1

    def write_to_file(self, path, iter):
        fname = f"{path}\\{self.split}_metrics.txt"
        if not os.path.isfile(fname):
            with open(fname, "a") as f:
                f.write(f"iter\tAVG_PSNR\tAVG_SSIM\tAVG_LPIPS\n")

        with open(fname, "a") as f:
            f.write(f"{iter}\t{self.avg_psnr[self.last_t]}\t{self.avg_ssim[self.last_t]}\t{self.avg_lpips[self.last_t]}\n")


def compute_perceptual_similarity(folder, pred_img, tgt_img, take_every_other):
    perceptual_loss = PerceptualLoss().eval().to("cuda")

    values_percsim = []
    values_ssim = []
    values_psnr = []
    folders = os.listdir(folder)
    for i, f in tqdm(enumerate(sorted(folders))):
        pred_imgs = glob.glob(folder + f + "/" + pred_img)
        tgt_imgs = glob.glob(folder + f + "/" + tgt_img)
        assert len(tgt_imgs) == 1

        perc_sim = 10000
        ssim_sim = -10
        psnr_sim = -10
        for p_img in pred_imgs:
            t_img = load_img(tgt_imgs[0])
            p_img = load_img(p_img, size=t_img.shape[2:])
            t_perc_sim = perceptual_sim(p_img, t_img, perceptual_loss).item()
            perc_sim = min(perc_sim, t_perc_sim)

            ssim_sim = max(ssim_sim, ssim_metric(p_img, t_img).item())
            psnr_sim = max(psnr_sim, psnr(p_img, t_img).item())

        values_percsim += [perc_sim]
        values_ssim += [ssim_sim]
        values_psnr += [psnr_sim]

    if take_every_other:
        n_valuespercsim = []
        n_valuesssim = []
        n_valuespsnr = []
        for i in range(0, len(values_percsim) // 2):
            n_valuespercsim += [min(values_percsim[2 * i], values_percsim[2 * i + 1])]
            n_valuespsnr += [max(values_psnr[2 * i], values_psnr[2 * i + 1])]
            n_valuesssim += [max(values_ssim[2 * i], values_ssim[2 * i + 1])]

        values_percsim = n_valuespercsim
        values_ssim = n_valuesssim
        values_psnr = n_valuespsnr

    avg_percsim = np.mean(np.array(values_percsim))
    std_percsim = np.std(np.array(values_percsim))

    avg_psnr = np.mean(np.array(values_psnr))
    std_psnr = np.std(np.array(values_psnr))

    avg_ssim = np.mean(np.array(values_ssim))
    std_ssim = np.std(np.array(values_ssim))

    return {
        "Perceptual similarity": (avg_percsim, std_percsim),
        "PSNR": (avg_psnr, std_psnr),
        "SSIM": (avg_ssim, std_ssim),
    }


def compute_perceptual_similarity_from_list(
    pred_imgs_list, tgt_imgs_list, take_every_other, simple_format=True
):
    # Load VGG16 for feature similarity
    perceptual_loss = PerceptualLoss().eval().to("cuda")

    values_percsim = []
    values_ssim = []
    values_psnr = []
    equal_count = 0
    ambig_count = 0
    for i, tgt_img in enumerate(tqdm(tgt_imgs_list)):
        pred_imgs = pred_imgs_list[i]
        tgt_imgs = [tgt_img]
        assert len(tgt_imgs) == 1

        if type(pred_imgs) != list:
            pred_imgs = [pred_imgs]

        perc_sim = 10000
        ssim_sim = -10
        psnr_sim = -10
        assert len(pred_imgs) > 0
        for p_img in pred_imgs:
            t_img = load_img(tgt_imgs[0])
            p_img = load_img(p_img, size=t_img.shape[2:])
            t_perc_sim = perceptual_sim(p_img, t_img, perceptual_loss).item()
            perc_sim = min(perc_sim, t_perc_sim)

            ssim_sim = max(ssim_sim, ssim_metric(p_img, t_img).item())
            psnr_sim = max(psnr_sim, psnr(p_img, t_img).item())

        values_percsim += [perc_sim]
        values_ssim += [ssim_sim]
        if psnr_sim != np.float("inf"):
            values_psnr += [psnr_sim]
        else:
            if torch.allclose(p_img, t_img):
                equal_count += 1
                print("{} equal src and wrp images.".format(equal_count))
            else:
                ambig_count += 1
                print("{} ambiguous src and wrp images.".format(ambig_count))

    if take_every_other:
        n_valuespercsim = []
        n_valuesssim = []
        n_valuespsnr = []
        for i in range(0, len(values_percsim) // 2):
            n_valuespercsim += [min(values_percsim[2 * i], values_percsim[2 * i + 1])]
            n_valuespsnr += [max(values_psnr[2 * i], values_psnr[2 * i + 1])]
            n_valuesssim += [max(values_ssim[2 * i], values_ssim[2 * i + 1])]

        values_percsim = n_valuespercsim
        values_ssim = n_valuesssim
        values_psnr = n_valuespsnr

    avg_percsim = np.mean(np.array(values_percsim))
    std_percsim = np.std(np.array(values_percsim))

    avg_psnr = np.mean(np.array(values_psnr))
    std_psnr = np.std(np.array(values_psnr))

    avg_ssim = np.mean(np.array(values_ssim))
    std_ssim = np.std(np.array(values_ssim))

    if simple_format:
        # just to make yaml formatting readable
        return {
            "Perceptual similarity": [float(avg_percsim), float(std_percsim)],
            "PSNR": [float(avg_psnr), float(std_psnr)],
            "SSIM": [float(avg_ssim), float(std_ssim)],
        }
    else:
        return {
            "Perceptual similarity": (avg_percsim, std_percsim),
            "PSNR": (avg_psnr, std_psnr),
            "SSIM": (avg_ssim, std_ssim),
        }

