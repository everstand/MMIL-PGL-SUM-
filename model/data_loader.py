# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
import random


def _inner_train_val_keys(train_keys, split_index, val_ratio=0.2, val_seed=0):
    """Deterministically carve inner validation keys from outer train_keys."""
    train_keys = list(train_keys)
    if len(train_keys) < 2:
        raise ValueError('At least two train_keys are required for inner validation.')
    val_count = max(1, int(round(len(train_keys) * val_ratio)))
    val_count = min(val_count, len(train_keys) - 1)
    shuffled = list(train_keys)
    random.Random(val_seed + split_index * 1009).shuffle(shuffled)
    val_set = set(shuffled[:val_count])
    inner_train_keys = [key for key in train_keys if key not in val_set]
    val_keys = [key for key in train_keys if key in val_set]
    return inner_train_keys, val_keys


class VideoData(Dataset):
    def __init__(self, mode, video_type, split_index, use_val_split=False,
                 val_ratio=0.2, val_seed=0):
        """Custom Dataset wrapper for frame features and ground-truth scores.

        Outer train/test keys always come from the official split JSON. When
        use_val_split=True, validation is carved only from outer train_keys.
        """
        self.mode = mode.lower()
        if self.mode not in ('train', 'val', 'test'):
            raise ValueError("mode must be one of: train, val, test.")
        self.name = video_type.lower()
        self.datasets = ['../PGL-SUM/data/datasets/SumMe/eccv16_dataset_summe_google_pool5.h5',
                         '../PGL-SUM/data/datasets/TVSum/eccv16_dataset_tvsum_google_pool5.h5']
        self.splits_filename = ['../PGL-SUM/data/datasets/splits/' + self.name + '_splits.json']
        self.split_index = split_index
        self.use_val_split = use_val_split
        self.val_ratio = val_ratio
        self.val_seed = val_seed

        if 'summe' in self.splits_filename[0]:
            self.filename = self.datasets[0]
        elif 'tvsum' in self.splits_filename[0]:
            self.filename = self.datasets[1]

        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split
                    break
            else:
                raise ValueError(f'split_index {self.split_index} not found.')

        outer_train_keys = list(self.split['train_keys'])
        test_keys = list(self.split['test_keys'])
        if use_val_split or self.mode == 'val':
            inner_train_keys, val_keys = _inner_train_val_keys(
                outer_train_keys, split_index, val_ratio, val_seed)
        else:
            inner_train_keys, val_keys = outer_train_keys, []

        self.key_sets = {
            'outer_train': outer_train_keys,
            'train': inner_train_keys,
            'val': val_keys,
            'test': test_keys,
        }
        self._check_disjoint()
        self.dataset_keys = self.key_sets[self.mode]

        self.list_frame_features, self.list_gtscores = [], []
        with h5py.File(self.filename, 'r') as hdf:
            for video_name in self.dataset_keys:
                frame_features = torch.Tensor(np.array(hdf[video_name + '/features']))
                gtscore = torch.Tensor(np.array(hdf[video_name + '/gtscore']))
                self.list_frame_features.append(frame_features)
                self.list_gtscores.append(gtscore)

    def _check_disjoint(self):
        train_keys = set(self.key_sets['train'])
        val_keys = set(self.key_sets['val'])
        test_keys = set(self.key_sets['test'])
        if train_keys & val_keys:
            raise ValueError('inner train_keys and val_keys overlap.')
        if train_keys & test_keys:
            raise ValueError('inner train_keys and outer test_keys overlap.')
        if val_keys & test_keys:
            raise ValueError('inner val_keys and outer test_keys overlap.')

    def __len__(self):
        self.len = len(self.dataset_keys)
        return self.len

    def __getitem__(self, index):
        video_name = self.dataset_keys[index]
        frame_features = self.list_frame_features[index]
        gtscore = self.list_gtscores[index]
        if self.mode in ('val', 'test'):
            return frame_features, video_name
        return frame_features, gtscore


def get_loader(mode, video_type, split_index, use_val_split=False,
               val_ratio=0.2, val_seed=0):
    vd = VideoData(mode, video_type, split_index, use_val_split=use_val_split,
                   val_ratio=val_ratio, val_seed=val_seed)
    if mode.lower() == 'train':
        return DataLoader(vd, batch_size=1, shuffle=True)
    return vd


if __name__ == '__main__':
    pass
