import pickle

from torch.utils.data import Dataset
import torchvision.transforms as T
import torch


def _check_available_pickle_file(data: dict, needed_keys: list[str]):
    """Check if the pickle file contains the necessary keys."""
    if not all([key in data for key in needed_keys]):
        raise ValueError(
            'The pickle file does not contain the necessary keys .'
            f'Needed keys: {needed_keys}, but got {list(data.keys())}.')


class TrainDataset(Dataset):

    def __init__(self, pkl_file, transform=None, target_transform=None):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        _check_available_pickle_file(data, ['images', 'labels'])

        # shape: (num_of_data, channel, width, height), range: [0.0, 1.0]
        self.images = torch.tensor(data['images'], dtype=torch.float32)
        # shape: (num_of_data,), range: [0, num_of_classes - 1]
        self.labels = torch.tensor(data['labels'], dtype=torch.long)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class InferDataset(Dataset):

    def __init__(self, pkl_file, transform=None):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        _check_available_pickle_file(
            data, ['sup_images', 'sup_labels', 'qry_images'])

        # shape: (num_of_task, num_of_data, channel, width, height), range: [0.0, 1.0]
        self.sup_images = torch.tensor(data['sup_images'], dtype=torch.float32)
        self.qry_images = torch.tensor(data['qry_images'], dtype=torch.float32)
        # shape: (num_of_task, num_of_data), range: [0, num_of_classes - 1]
        self.sup_labels = torch.tensor(data['sup_labels'], dtype=torch.long)

        self.transform = transform

    def __len__(self):
        return len(self.sup_images)

    def __getitem__(self, idx):
        sup_image = self.sup_images[idx]
        qry_image = self.qry_images[idx]
        sup_label = self.sup_labels[idx]

        if self.transform:
            sup_image = self.transform(sup_image)
            qry_image = self.transform(qry_image)

        return sup_image, sup_label, qry_image