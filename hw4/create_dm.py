import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn
from torch.utils.data import Dataset
import torchvision
from torchvision import io
import torchvision.transforms.functional as TF
from tqdm import tqdm
from tqdm import trange


def _compute_density_map(points: tuple[torch.Tensor, torch.Tensor],
                         size: tuple[int, int],
                         kernel_size=101,
                         sigma=15.0,
                         k=5,
                         dynamic_sigma=False) -> torch.Tensor:
    """Compute the density map of the given points.

    Args:
        points: List of points, each point is a tuple of (h, w).
        size: Size of the density map, (H, W).
        kernel_size: Size of the Gaussian kernel.
        sigma: Sigma of the Gaussian kernel. Only used when `dynamic_sigma` is
            False.
        k: Number of nearest neighbors to use when computing the sigma of each
            point. Only used when `dynamic_sigma` is True.
        dynamic_sigma: Whether to compute the sigma of each point dynamically.

    Returns:
        The density map.
    """
    density_map = torch.zeros((1, *size), dtype=torch.double, device=DEVICE)

    hs, ws = points
    if torch.numel(hs) == 0 or torch.numel(ws) == 0:
        return density_map

    hs.clip_(0, size[0] - 1)
    ws.clip_(0, size[1] - 1)

    if dynamic_sigma:
        pts = torch.stack((hs, ws), dim=-1).double()
        dist = torch.cdist(pts, pts, p=2)

        if torch.numel(dist) == 1:
            sigmas = [size[0] * size[1] / 4.0]
        else:
            k = min(k, torch.numel(dist) - 1)
            k_nearest, _ = torch.topk(dist, k + 1, dim=-1, largest=False)
            # remove self (the closest point is self)
            k_nearest = k_nearest[:, 1:]
            sigmas = (0.3 * torch.mean(k_nearest, dim=-1)).tolist()

        density_maps = []
        for h, w, sig in tqdm(zip(hs, ws, sigmas),
                              total=len(hs),
                              leave=False,
                              dynamic_ncols=True):
            pt2d = torch.zeros((1, *size), dtype=torch.double, device=DEVICE)
            pt2d[:, h, w] = 1.0
            density_maps.append(TF.gaussian_blur(pt2d, kernel_size, sig))

            if len(density_maps) > 16:
                density_map += torch.cat(density_maps, dim=0).sum(dim=0,
                                                                  keepdim=True)
                density_maps.clear()

        if density_maps:
            density_map += torch.cat(density_maps, dim=0).sum(dim=0,
                                                              keepdim=True)

    else:
        density_map[:, hs, ws] = 1.0
        density_map = TF.gaussian_blur(density_map, kernel_size, sigma)

    return density_map


class PreprocessDataset(Dataset):

    def __init__(self, dir_path, transform=None, target_transform=None):

        self.dir_path = dir_path
        self.filenames = [f.stem for f in Path(dir_path).glob('*.jpg')]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dir_path, f'{self.filenames[idx]}.jpg')
        label_path = os.path.join(self.dir_path, f'{self.filenames[idx]}.npy')

        # shape: (C, H, W), range: [0, 255]
        image = io.read_image(image_path)
        image = image.type(torch.double) / 255.0

        label = torch.tensor(np.load(label_path),
                             dtype=torch.long,
                             device=image.device)
        if label.size(0) > 0:
            ws, hs = torch.unbind(label, dim=-1)
        else:
            ws, hs = torch.tensor([]), torch.tensor([])

        return image, hs, ws, self.filenames[idx]


def main():
    ds = PreprocessDataset(args.input_dir)

    # image, hs, ws, name = ds[10]
    # dm = _compute_density_map(
    #     (hs, ws),
    #     image.shape[1:],
    #     k=5,
    #     dynamic_sigma=True,
    # )
    # torchvision.utils.save_image((dm * 255).clip(0, 255),
    #                              os.path.join(f'dm.jpg'))
    # torchvision.utils.save_image((dm * 50 + image.to(DEVICE)).clip(0, 255),
    #                              os.path.join(f'dm1.jpg'))

    with trange(len(ds), dynamic_ncols=True) as pbar:
        for i in pbar:
            image, hs, ws, name = ds[i]
            dm = _compute_density_map(
                (hs, ws),
                image.shape[1:],
                sigma=3,
                # k=3,
                # dynamic_sigma=args.dynamic,
            )
            torchvision.utils.save_image(
                (dm * 255).clip(0, 255),
                os.path.join(args.output_dir, f'{name}.jpg'),
            )

            pbar.set_postfix_str(f'gt={len(hs)}, dm={dm.sum().item():.4f}')


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input-dir',
        default=os.path.join('data_processed', 'train'),
        type=str,
        help=
        'Path to the directory containing the images(.jpg) and labels(.npy).')
    parser.add_argument('--output-dir',
                        default=os.path.join('data', 'train', 'dm3'),
                        type=str,
                        help='Path to the output directory.')
    parser.add_argument('--postfix',
                        default='',
                        type=str,
                        help='Postfix of the density map.')
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Whether to compute the sigma of each point dynamically.')
    args = parser.parse_args()

    main()
