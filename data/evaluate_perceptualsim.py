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
    try:
        img = Image.open(img_name)

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

        ssim_val = self.compute_ssim(pred_img, tgt_img, batch)
        psnr_val = self.compute_psnr(pred_img, tgt_img, batch)

    def calc_average(self, t):
        psnr_val = 0
        ssim_val = 0

        for batch in self.batches:
            psnr_val += self.values_psnr[batch][t]
            ssim_val += self.values_ssim[batch][t]

        psnr_avg = psnr_val/len(self.batches)
        self.avg_psnr.append(psnr_avg)

        ssim_avg = ssim_val/len(self.batches)
        self.avg_ssim.append(ssim_avg)

        print(f"Avg PSNR: {self.avg_psnr[t]} - Avg SSIM: {self.avg_ssim[t]}")



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


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--folder", type=str, default="")
    args.add_argument("--pred_image", type=str, default="")
    args.add_argument("--target_image", type=str, default="")
    args.add_argument("--take_every_other", action="store_true", default=False)
    args.add_argument("--output_file", type=str, default="")

    opts = args.parse_args()

    folder = opts.folder
    pred_img = opts.pred_image
    tgt_img = opts.target_image

    results = compute_perceptual_similarity(
        folder, pred_img, tgt_img, opts.take_every_other
    )

    f = open(opts.output_file, "w")
    for key in results:
        print("%s for %s: \n" % (key, opts.folder))
        print("\t {:0.4f} | {:0.4f} \n".format(results[key][0], results[key][1]))

        f.write("%s for %s: \n" % (key, opts.folder))
        f.write("\t {:0.4f} | {:0.4f} \n".format(results[key][0], results[key][1]))

    f.close()
