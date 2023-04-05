import torch


def euclidean_distance_sqaured(a: torch.Tensor, b: torch.Tensor):
    """Compute euclidean distance between two tensors.

    Args:
        a: Tensor of shape (n_samples, emb_size).
        b: Tensor of shape (n_classes, emb_size).
    Returns:
        Tensor of shape (n_samples, n_classes).
    """
    n_samples = a.shape[0]
    n_classes = b.shape[0]
    a = a.unsqueeze(1).expand(n_samples, n_classes, -1)
    b = b.unsqueeze(0).expand(n_samples, n_classes, -1)
    dist_square = ((a - b)**2).sum(dim=-1)
    return dist_square
