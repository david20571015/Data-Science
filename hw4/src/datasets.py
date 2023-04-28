import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import io
import torchvision.transforms.functional as TF


class TrainDataset(Dataset):

    def __init__(self,
                 image_dir,
                 gt_dir,
                 transforms=None,
                 transform=None,
                 target_transform=None):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
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

        gt_path = os.path.join(self.gt_dir, f'{self.filenames[idx]}.jpg')
        # shape: (1, H, W), range: [0, 255]
        density_map = io.read_image(gt_path, io.ImageReadMode.GRAY)
        # map to [0.0, 1.0]
        density_map = TF.convert_image_dtype(density_map, dtype=torch.float)

        if self.transforms:
            transformed_images = self.transforms(
                torch.cat((image, density_map)))
            image, density_map = torch.split(
                transformed_images,
                [image.size(0), density_map.size(0)],
                dim=0)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            density_map = self.target_transform(density_map)

        return image, density_map


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

        # image = four_crop(image)

        return file_id, image
