from typing import Optional

import torch
import torch.nn.functional as F


def bayesian_loss(predict_dm: torch.Tensor,
                  post_probs: Optional[torch.Tensor]):
    if post_probs is None:
        count = predict_dm.sum()
        target = torch.zeros_like(count)
    else:
        count = (predict_dm * post_probs).sum((-1, -2))
        target = torch.ones_like(count)

    return F.l1_loss(count, target, reduction='sum')


def compute_post_probs(gt: torch.Tensor, sigma=8.0, stride=8):
    cood = torch.arange(
        0.0,
        gt.size(-1),
        stride,
        dtype=torch.float,
        device=gt.device,
    ) + stride / 2.0

    cood.unsqueeze_(0)

    x_idx, y_idx = torch.nonzero(gt.squeeze(), as_tuple=True)
    x_idx = x_idx.float().unsqueeze(1)
    y_idx = y_idx.float().unsqueeze(1)

    if x_idx.numel() == 0:
        return None

    x_dis = x_idx**2 - 2 * x_idx @ cood + cood**2
    y_dis = y_idx**2 - 2 * y_idx @ cood + cood**2

    dis = x_dis.unsqueeze(2) + y_dis.unsqueeze(1)
    dis = -dis / (2.0 * sigma**2)

    C, H, W = dis.size()

    return torch.softmax(dis.view(C, -1), dim=-1).view(C, H, W)


def compute_loss(pred: torch.Tensor, gt: torch.Tensor, sigma=8.0):
    post_probs = compute_post_probs(gt, sigma=sigma)
    loss = bayesian_loss(pred, post_probs)
    return loss
