from typing import Optional

import torch
import torch.nn.functional as F


def compute_post_probs(points: torch.Tensor,
                       short_size: torch.Tensor,
                       sigma=8.0,
                       stride=8,
                       use_bg=False):
    if points.numel() == 0:
        return None

    cood = torch.arange(
        0.0,
        512.0,
        stride,
        dtype=torch.float,
        device=points.device,
    ) + stride / 2.0

    cood.unsqueeze_(0)

    x_idx, y_idx = torch.unbind(points, dim=-1)
    x_idx = x_idx.float().unsqueeze(1)
    y_idx = y_idx.float().unsqueeze(1)

    # x_dis = x_idx**2 - 2 * x_idx @ cood + cood**2
    x_dis = (x_idx - cood)**2
    # y_dis = y_idx**2 - 2 * y_idx @ cood + cood**2
    y_dis = (y_idx - cood)**2

    dis = x_dis.unsqueeze(1) + y_dis.unsqueeze(2)

    if use_bg:
        bg_dis = short_size**2 / (torch.min(dis, dim=0, keepdim=True).values +
                                  1e-5)
        dis = torch.cat((dis, bg_dis), dim=0)

    dis = -dis / (2.0 * sigma**2)

    C, H, W = dis.size()

    return torch.softmax(dis.view(C, -1), dim=-1).view(C, H, W)


def bayesian_loss(predict_dm: torch.Tensor,
                  target: torch.Tensor,
                  post_probs: Optional[torch.Tensor],
                  use_bg=False):
    if post_probs is None:
        count = predict_dm.sum()
        target = torch.zeros_like(count)
    else:
        if use_bg:
            target = torch.cat((target, torch.zeros_like(target[0:1])), dim=0)
        count = (predict_dm * post_probs).sum((-1, -2))

    return F.l1_loss(count, target)


def compute_loss(preds: torch.Tensor,
                 points: list[torch.Tensor],
                 targets: list[torch.Tensor],
                 short_sizes: torch.Tensor,
                 sigma=8.0):

    prob_list = [
        compute_post_probs(point, st_size, sigma=sigma)
        for point, st_size in zip(points, short_sizes)
    ]

    losses = [
        bayesian_loss(p, t, post_probs)
        for p, t, post_probs in zip(preds, targets, prob_list)
    ]

    return torch.mean(torch.stack(losses))
