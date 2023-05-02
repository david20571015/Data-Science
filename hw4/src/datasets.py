import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import io
import torchvision.transforms.functional as TF
import numpy as np


class TrainDataset(Dataset):

    def __init__(self,
                 image_dir,
                 transforms=None,
                 transform=None,
                 target_transform=None):
        self.image_dir = image_dir
        self.filenames = [
            f.stem for f in Path(image_dir).glob('*.jpg') if f.stem.isdigit()
        ]

        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, f'{self.filenames[idx]}.jpg')
        # shape: (C, H, W), range: [0, 255]
        image = io.read_image(image_path, io.ImageReadMode.RGB)
        # map to [0.0, 1.0]
        image = TF.convert_image_dtype(image, dtype=torch.float)

        label_path = os.path.join(self.image_dir, f'{self.filenames[idx]}.npy')
        label = torch.tensor(np.load(label_path),
                             dtype=torch.long,
                             device=image.device)

        gt = torch.zeros((1, *image.shape[-2:]),
                         dtype=torch.float,
                         device=image.device)

        if label.size(0) > 0:
            ws, hs = torch.unbind(label, dim=-1)
            ws.clamp_(0, image.shape[-1] - 1)
            hs.clamp_(0, image.shape[-2] - 1)

            gt[0, hs, ws] = 1.0

        if self.transforms:
            transformed_images = self.transforms(torch.cat((image, gt)))
            image, gt = torch.split(transformed_images,
                                    [image.size(0), gt.size(0)],
                                    dim=0)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            gt = self.target_transform(gt)

        return image, gt


def four_crop(image: torch.Tensor):
    *_, H, W = image.size()
    (top_left, top_right, bottom_left, bottom_right,
     _) = TF.five_crop(image, size=(H // 2, W // 2))
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
