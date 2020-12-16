from typing import Optional

import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler

from scarches.dataset.mars import MetaAnnotatedDataset
from scarches.trainers.trvae._utils import _print_progress_bar


def print_meta_progress(epoch, logs, n_epochs=10000):
    """Creates Message for '_print_progress_bar'.

       Parameters
       ----------
       epoch: Integer
            Current epoch iteration.
       logs: dict
            List of all current losses.
       n_epochs: Integer
            Maximum value of epochs.

       Returns
       -------
    """
    message = f' - loss: {logs["loss"][-1]:.4f}'
    train_keys = [key for key in sorted(list(logs.keys())) if
                  (not key.startswith('val_') and not key.startswith('test_') and key != 'loss')]

    for key in train_keys:
        message += f' - {key}: {logs[key][-1]:.4f}'

    message += f' - val_loss: {logs["val_loss"][-1]:.4f}'
    valid_keys = [key for key in sorted(list(logs.keys())) if (key.startswith('val_') and key != 'val_loss')]

    for key in valid_keys:
        message += f' - {key}: {logs[key][-1]:.4f}'

    test_keys = [key for key in sorted(list(logs.keys())) if (key.startswith('test_'))]
    for key in test_keys:
        message += f' - {key}: {logs[key][-1]:.4f}'

    _print_progress_bar(epoch + 1, n_epochs, prefix='', suffix=message, decimals=1, length=20)


def split_meta_train_tasks(meta_adata: MetaAnnotatedDataset,
                           train_fraction: float = 0.8,
                           stratify: bool = False,
                           batch_size: Optional[int] = 32,
                           num_workers=6):
    train_data_loaders, valid_data_loaders = [], []

    for idx, meta_train_dataset in enumerate(meta_adata.meta_train_tasks_adata):
        np.random.seed(2020 * idx)
        n_samples = len(meta_train_dataset)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        train_idx = indices[:int(train_fraction * n_samples)]
        valid_idx = indices[int(train_fraction * n_samples):]

        if batch_size < 0:
            train_size = len(train_idx)
            valid_size = len(valid_idx)
        else:
            train_size = valid_size = batch_size

        train_data_loader = DataLoader(dataset=meta_train_dataset,
                                       sampler=SubsetRandomSampler(train_idx),
                                       num_workers=num_workers,
                                       batch_size=train_size,
                                       pin_memory=True)

        valid_data_loader = DataLoader(dataset=meta_train_dataset,
                                       sampler=SubsetRandomSampler(valid_idx),
                                       num_workers=num_workers,
                                       batch_size=valid_size,
                                       pin_memory=True)

        train_data_loaders.append(train_data_loader)
        valid_data_loaders.append(valid_data_loader)

    return train_data_loaders, valid_data_loaders


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
