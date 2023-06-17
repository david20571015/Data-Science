from typing import Literal

import lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader


class BaseDataset(InMemoryDataset):

    def __init__(self, root='data'):
        super().__init__(root)

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def raw_dir(self):
        return self.root


class TrainDataset(BaseDataset):

    def __init__(self,
                 root='data',
                 train_ratio=0.8,
                 stage: Literal['train', 'valid'] = 'train'):
        self.train_ratio = train_ratio
        super().__init__(root)

        self._data = torch.load(self.processed_paths[0])

        # if stage == 'train':
        #     self._data = torch.load(self.processed_paths[0])
        # elif stage == 'valid':
        #     self._data = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return [
            'train_sub-graph_tensor.pt',
            'train_mask.npy',
        ]

    @property
    def processed_file_names(self):
        return [
            'processed_train_data.pt',
            # 'processed_valid_data.pt',
        ]

    def process(self):
        data: Data = torch.load(self.raw_paths[0])
        mask: np.ndarray = np.load(self.raw_paths[1])

        mask_idx = np.arange(len(mask))[mask]

        while True:
            train_mask_idx, valid_mask_idx = train_test_split(
                mask_idx, train_size=self.train_ratio)

            train_mask = torch.zeros(len(mask), dtype=torch.bool)
            train_mask[train_mask_idx] = True

            valid_mask = torch.zeros(len(mask), dtype=torch.bool)
            valid_mask[valid_mask_idx] = True

            y = torch.zeros(len(mask), dtype=torch.double)
            y[mask] = data.label.double()

            # Make sure that both train and valid set have positive and negative
            if 0 < y[train_mask].mean() < 1 and 0 < y[valid_mask].mean() < 1:
                break

        data = Data(
            x=data.feature,
            edge_index=data.edge_index,
            y=y.unsqueeze(-1),
            train_mask=train_mask,
            valid_mask=valid_mask,
        )

        torch.save(data, self.processed_paths[0])

        # train_data = data.subgraph(train_mask)
        # torch.save(train_data, self.processed_paths[0])

        # valid_data = data.subgraph(valid_mask)
        # torch.save(valid_data, self.processed_paths[1])

        # return Data(
        #     x=data.feature,
        #     edge_index=data.edge_index,
        #     y=y.unsqueeze(-1),
        #     train_mask=train_mask,
        #     valid_mask=valid_mask,
        # )


class TestDataset(BaseDataset):

    def __init__(self, root='data'):
        super().__init__(root)
        self._data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            'test_sub-graph_tensor_noLabel.pt',
            'test_mask.npy',
        ]

    @property
    def processed_file_names(self):
        return [
            'processed_test_data.pt',
        ]

    def process(self):
        data: Data = torch.load(self.raw_paths[0])
        mask: np.ndarray = np.load(self.raw_paths[1])

        test_data = Data(
            x=data.feature,
            edge_index=data.edge_index,
            test_mask=mask,
        )

        torch.save(test_data, self.processed_paths[0])


class GraphDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = 'data', train_ratio=0.8):
        super().__init__()
        self.data_dir = data_dir
        self.train_ratio = train_ratio

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.dataloader_kwargs = {
            'shuffle': False,
            'batch_size': 1,
            'num_workers': 4,
            'pin_memory': True,
        }

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = TrainDataset(self.data_dir, self.train_ratio,
                                              'train')
            # self.valid_dataset = TrainDataset(self.data_dir, self.train_ratio,
            #                                 'valid')
        elif stage == 'predict':
            self.test_dataset = TestDataset(self.data_dir)
        else:
            raise ValueError(f'Invalid stage: {stage}')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, **self.dataloader_kwargs)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, **self.dataloader_kwargs)
