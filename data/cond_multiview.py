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

@dataclass
class MVDreamMultiviewsDataModuleConfig(MultiviewsDataModuleConfig):
    # Dataset parameters
    n_view: int = 4
    crop_to: int = 1024
    input_size: int = 256
    novel_frame_count: int = 1
    train_split: str = "train"

    # Random camera parameters
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

class MVDreamMultiviewDataset(Dataset):
   
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: MVDreamMultiviewsDataModuleConfig = cfg

        camera_dict = json.load(
            open(os.path.join(self.cfg.dataroot, f"transforms_{split}.json"), "r")
        )

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

        self.cfg.input_size = 256
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
            img = cv2.imread(frame_path)[:, :, ::-1].copy()
            
            img = crop_center(img, self.cfg.crop_to, self.cfg.crop_to)

            img = cv2.resize(img, (self.frame_w, self.frame_h))

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
        return self.frames_proj.shape[0]

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.frames_c2w[index],
            "camera_positions": self.frames_position[index],
            "light_positions": self.light_positions[index],
            "gt_rgb": self.frames_img[index],
        }

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.frame_h, "width": self.frame_w})
        return batch

class NovelFrames():
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

        positions = NovelFrames.fibonacci_northern_hemisphere(self.n_frames)
        autogen_sampled_poses = [NovelFrames.create_camera_to_world_matrix_fib(pos) for pos in positions]

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
            transparency_mask: Float[Tensor, "H W 3"] = torch.FloatTensor(transparency_mask == 0).unsqueeze(dim=-1) # Boolean transparency mask
            transparency_masks.append(transparency_mask)

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
        self.transparency_masks: Float[Tensor, "B H W 3"] = torch.stack(transparency_masks, dim=0)

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

        self.novel_frames = NovelFrames(self.cfg, camera_dict, 1000)

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        assert (
            self.batch_size % self.cfg.n_view == 0
        ), f"batch_size ({self.batch_size}) must be dividable by n_view ({self.cfg.n_view})!"
        real_batch_size = self.batch_size // self.cfg.n_view

        # Use 1 GT images from train set and 1 novel view in batch for guidance: [novel, gt, gt, gt]
        if self.cfg.novel_frame_count < self.cfg.n_view:
            train_indexes = torch.randperm(self.n_frames)[:(self.cfg.n_view-self.cfg.novel_frame_count)]
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

        if self.cfg.novel_frame_count != 0:
            novel_idx = torch.randint(self.novel_frames.n_frames, (self.cfg.novel_frame_count,))
        else:
            novel_idx = []

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
            "novel_frame_count": self.cfg.novel_frame_count,
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
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
            #self.val_dataset = MVDreamMultiviewDataset(self.cfg, "val16")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test")
            #self.test_dataset = MVDreamMultiviewDataset(self.cfg, "test")

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
