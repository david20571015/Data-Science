import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import io
import torchvision.transforms.functional as TF


def random_crop(im_h: int, im_w: int, crop_h: int, crop_w: int):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = np.random.default_rng(0).integers(0, res_h + 1)
    j = np.random.default_rng(0).integers(0, res_w + 1)
    return i, j, crop_h, crop_w


def cal_innner_area(left: int, top: int, right: int, bottom: int,
                    bbox: torch.Tensor):
    b_left, b_top, b_right, b_bottom = bbox.unbind(dim=-1)
    inner_left = torch.max(torch.tensor(left, device=bbox.device), b_left)
    inner_right = torch.min(torch.tensor(right, device=bbox.device), b_right)
    inner_top = torch.max(torch.tensor(top, device=bbox.device), b_top)
    inner_bottom = torch.min(torch.tensor(bottom, device=bbox.device), b_bottom)
    inner_area = (torch.clamp(inner_right - inner_left, min=0) *
                  torch.clamp(inner_bottom - inner_top, min=0))
    return inner_area


class TrainDataset(Dataset):

    def __init__(self,
                 image_dir,
                 crop_size=512,
                 transforms=None,
                 transform=None,
                 target_transform=None):
        self.image_dir = image_dir
        self.filenames = [
            f.stem for f in Path(image_dir).glob('*.jpg') if f.stem.isdigit()
        ]
        self.crop_size = crop_size

        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # image
        image_path = os.path.join(self.image_dir, f'{self.filenames[idx]}.jpg')
        # shape: (C, H, W), range: [0, 255]
        image = io.read_image(image_path, io.ImageReadMode.RGB)
        # map to [0.0, 1.0]
        image = TF.convert_image_dtype(image, dtype=torch.float)

        _, H, W = image.shape
        short_size = min(H, W)

        top, left, height, width = random_crop(H, W, self.crop_size,
                                               self.crop_size)
        image = TF.crop(image, top, left, height, width)

        # keypoint
        label_path = os.path.join(self.image_dir, f'{self.filenames[idx]}.npy')
        label = torch.tensor(np.load(label_path),
                             dtype=torch.long,
                             device=image.device)
        if label.size(0) > 0:
            nearest_dis = torch.clamp(label[..., 2], 4.0, 128.0)
            left_top = label[..., :2] - nearest_dis[..., None] / 2.0
            right_bottom = label[..., :2] + nearest_dis[..., None] / 2.0

            bbox = torch.cat((left_top, right_bottom), dim=-1)
            inner_area = cal_innner_area(left, top, left + width, top + height,
                                         bbox)

            origin_area = nearest_dis**2

            ratio = torch.clamp((inner_area / origin_area), 0.0, 1.0)

            mask = ((ratio >= 0.2) & (left <= label[:, 0]) &
                    (label[:, 0] < left + width) & (top <= label[:, 1]) &
                    (label[:, 1] < top + height))

            keypoints = label[mask][..., :2]
            keypoints[:, 0] -= left
            keypoints[:, 1] -= top

            target = ratio[mask]
        else:
            keypoints = torch.zeros((0, 2),
                                    dtype=torch.float,
                                    device=image.device)
            target = torch.zeros((0,), dtype=torch.float, device=image.device)

        if random.random() > 0.5:
            image = TF.hflip(image)
            if keypoints.size(0) > 0:
                keypoints[..., 0] = (width - 1) - keypoints[..., 0]

        if self.transform:
            image = self.transform(image)

        return image, keypoints.float(), target.float(), short_size


def four_crop(image: torch.Tensor):
    *_, H, W = image.size()
    (top_left, top_right, bottom_left, bottom_right,
     _) = TF.five_crop(image, size=[H // 2, W // 2])
    return torch.stack([top_left, top_right, bottom_left, bottom_right])


class InferDataset(Dataset):

    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.filenames = [f.stem for f in Path(dir_path).glob('*.jpg')]

        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_id = torch.tensor(int(self.filenames[idx]), dtype=torch.long)

        image_path = os.path.join(self.dir_path, f'{self.filenames[idx]}.jpg')
        # shape: (C, H, W), range: [0, 255]
        image = io.read_image(image_path, io.ImageReadMode.RGB)
        # map to [0.0, 1.0]
        image = TF.convert_image_dtype(image, dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return file_id, image
