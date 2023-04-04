import torch

import random


def _random_sample(datas, n_samples: int) -> list:
    idx = torch.randperm(len(datas))[:n_samples]
    return datas[idx]


class CategoriesSampler:

    def __init__(
        self,
        labels: torch.Tensor,
        n_classes: int,
        n_batch: int,
        n_way: int,
        n_shot: int,
        n_query: int,
    ):
        self.labels = labels
        self.n_classes = n_classes
        self.n_batch = n_batch
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        self.class_to_index = [
            torch.nonzero(labels == i).view(-1) for i in range(n_classes)
        ]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        """
        Yields:
            indices: shape: (n_way, n_shot + n_query).
        """
        for _ in range(self.n_batch):
            classes = random.sample(range(self.n_classes), self.n_way)

            batches = []

            for c in classes:
                selected = _random_sample(self.class_to_index[c],
                                          self.n_shot + self.n_query)
                batches.append(selected)

            yield batches
