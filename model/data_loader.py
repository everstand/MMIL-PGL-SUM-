# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
import random
from pathlib import Path


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


def _load_weak_label_payload(weak_labels_path):
    weak_labels_path = Path(weak_labels_path)
    if not weak_labels_path.exists():
        raise FileNotFoundError(f'Weak label file not found: {weak_labels_path}')
    payload = np.load(weak_labels_path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise TypeError(f'Weak label file must contain a dict: {weak_labels_path}')
    return payload


def _validate_finite(name, array):
    array = np.asarray(array)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} contains NaN or Inf values.')


def _expand_shot_utility_to_features(video_name, video_payload, hdf):
    if not isinstance(video_payload, dict):
        raise TypeError(f'Weak label payload for {video_name} must be a dict.')
    if 'shot_utility' not in video_payload:
        raise KeyError(f'Missing shot_utility for {video_name}.')

    shot_utility = np.asarray(video_payload['shot_utility'], dtype=np.float32)
    event_valid_mask = np.asarray(
        video_payload.get('event_valid_mask', np.ones_like(shot_utility, dtype=bool)),
        dtype=bool,
    )

    frame_features = np.asarray(hdf[video_name + '/features'])
    picks = np.asarray(hdf[video_name + '/picks'])
    change_points = np.asarray(hdf[video_name + '/change_points'])

    if shot_utility.ndim != 1:
        raise ValueError(f'shot_utility for {video_name} must be 1D.')
    if event_valid_mask.ndim != 1:
        raise ValueError(f'event_valid_mask for {video_name} must be 1D.')
    if len(shot_utility) != len(change_points):
        raise ValueError(
            f'shot_utility length mismatch for {video_name}: '
            f'{len(shot_utility)} vs {len(change_points)} change points.'
        )
    if len(event_valid_mask) != len(change_points):
        raise ValueError(
            f'event_valid_mask length mismatch for {video_name}: '
            f'{len(event_valid_mask)} vs {len(change_points)} change points.'
        )

    shot_end = change_points[:, 1]
    shot_index = np.searchsorted(shot_end, picks, side='left')
    shot_index = np.clip(shot_index, 0, len(shot_utility) - 1)
    starts = change_points[shot_index, 0]
    ends = change_points[shot_index, 1]
    if np.any((picks < starts) | (picks > ends)):
        raise ValueError(f'Failed to align weak shot utility to sampled frames for {video_name}.')

    weak_target = shot_utility[shot_index].astype(np.float32)
    weak_mask = event_valid_mask[shot_index].astype(bool)

    if weak_target.shape[0] != frame_features.shape[0]:
        raise ValueError(
            f'Weak target length mismatch for {video_name}: '
            f'{weak_target.shape[0]} vs features {frame_features.shape[0]}.'
        )
    _validate_finite(f'{video_name} weak_target', weak_target)
    if int(weak_mask.sum()) <= 0:
        raise ValueError(f'{video_name} weak_mask.sum() must be > 0.')
    return weak_target, weak_mask


def _build_pos_neg_masks_from_weak_target(weak_target, weak_mask, pos_ratio=0.15, neg_ratio=0.15):
    weak_target = np.asarray(weak_target, dtype=np.float32)
    weak_mask = np.asarray(weak_mask, dtype=bool)
    labeled_idx = np.flatnonzero(weak_mask)
    if labeled_idx.size == 0:
        raise ValueError('weak_mask must include at least one labeled position.')

    pos_mask = np.zeros_like(weak_mask, dtype=bool)
    neg_mask = np.zeros_like(weak_mask, dtype=bool)
    labeled_scores = weak_target[labeled_idx]

    pos_count = max(1, int(round(labeled_idx.size * pos_ratio)))
    neg_count = max(1, int(round(labeled_idx.size * neg_ratio)))
    if labeled_idx.size == 1:
        pos_count, neg_count = 1, 0
    else:
        pos_count = min(pos_count, labeled_idx.size - 1)
        neg_count = min(neg_count, labeled_idx.size - pos_count)
        if neg_count <= 0:
            neg_count = 1
            pos_count = max(1, labeled_idx.size - 1)

    order = np.argsort(labeled_scores)
    if neg_count > 0:
        neg_idx = labeled_idx[order[:neg_count]]
        neg_mask[neg_idx] = True
    if pos_count > 0:
        pos_idx = labeled_idx[order[-pos_count:]]
        pos_mask[pos_idx] = True

    overlap = pos_mask & neg_mask
    if overlap.any():
        neg_mask[overlap] = False
    return pos_mask, neg_mask


class VideoData(Dataset):
    def __init__(self, mode, video_type, split_index, use_val_split=False,
                 val_ratio=0.2, val_seed=0, dataset_root=None, splits_root=None,
                 supervision_setting='supervised', weak_labels_path=None,
                 weak_pos_ratio=0.15, weak_neg_ratio=0.15):
        """Custom Dataset wrapper for frame features and train/eval targets."""
        self.mode = mode.lower()
        if self.mode not in ('train', 'val', 'test'):
            raise ValueError('mode must be one of: train, val, test.')

        self.name = video_type.lower()
        self.split_index = split_index
        self.use_val_split = use_val_split
        self.val_ratio = val_ratio
        self.val_seed = val_seed
        self.supervision_setting = supervision_setting.lower()
        self.weak_pos_ratio = weak_pos_ratio
        self.weak_neg_ratio = weak_neg_ratio
        self.weak_labels_path = Path(weak_labels_path) if weak_labels_path is not None else None

        dataset_root = Path(dataset_root)
        splits_root = Path(splits_root)

        dataset_dir = 'SumMe' if self.name == 'summe' else 'TVSum'
        self.filename = dataset_root / dataset_dir / f'eccv16_dataset_{self.name}_google_pool5.h5'
        self.splits_filename = splits_root / f'{self.name}_splits.json'

        if not self.filename.exists():
            raise FileNotFoundError(f'Dataset file not found: {self.filename}')
        if not self.splits_filename.exists():
            raise FileNotFoundError(f'Split file not found: {self.splits_filename}')

        with open(self.splits_filename) as f:
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

        self.list_frame_features = []
        self.train_samples = []
        weak_label_data = None
        if self.mode == 'train' and self.supervision_setting == 'weak':
            if self.weak_labels_path is None:
                raise ValueError('weak mode requires weak_labels_path.')
            weak_label_data = _load_weak_label_payload(self.weak_labels_path)
            missing = sorted(set(self.dataset_keys) - set(weak_label_data.keys()))
            if missing:
                raise KeyError(f'Missing weak labels for train keys: {missing}')

        with h5py.File(self.filename, 'r') as hdf:
            for video_name in self.dataset_keys:
                frame_features = torch.tensor(np.asarray(hdf[video_name + '/features']), dtype=torch.float32)
                self.list_frame_features.append(frame_features)

                if self.mode != 'train':
                    continue

                if self.supervision_setting == 'weak':
                    weak_target_np, weak_mask_np = _expand_shot_utility_to_features(
                        video_name, weak_label_data[video_name], hdf)
                    pos_mask_np, neg_mask_np = _build_pos_neg_masks_from_weak_target(
                        weak_target_np, weak_mask_np,
                        pos_ratio=self.weak_pos_ratio,
                        neg_ratio=self.weak_neg_ratio,
                    )
                    self.train_samples.append({
                        'frame_features': frame_features,
                        'target': torch.tensor(weak_target_np, dtype=torch.float32),
                        'mask': torch.tensor(weak_mask_np, dtype=torch.bool),
                        'pos_mask': torch.tensor(pos_mask_np, dtype=torch.bool),
                        'neg_mask': torch.tensor(neg_mask_np, dtype=torch.bool),
                        'video_name': video_name,
                    })
                else:
                    gtscore = torch.tensor(np.asarray(hdf[video_name + '/gtscore']), dtype=torch.float32)
                    self.train_samples.append({
                        'frame_features': frame_features,
                        'target': gtscore,
                        'mask': torch.ones_like(gtscore, dtype=torch.bool),
                        'pos_mask': torch.zeros_like(gtscore, dtype=torch.bool),
                        'neg_mask': torch.zeros_like(gtscore, dtype=torch.bool),
                        'video_name': video_name,
                    })

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
        return len(self.dataset_keys)

    def __getitem__(self, index):
        video_name = self.dataset_keys[index]
        frame_features = self.list_frame_features[index]
        if self.mode in ('val', 'test'):
            return frame_features, video_name
        return self.train_samples[index]


def get_loader(mode, video_type, split_index, use_val_split=False,
               val_ratio=0.2, val_seed=0, dataset_root=None, splits_root=None,
               supervision_setting='supervised', weak_labels_path=None,
               weak_pos_ratio=0.15, weak_neg_ratio=0.15):
    vd = VideoData(
        mode, video_type, split_index,
        use_val_split=use_val_split,
        val_ratio=val_ratio,
        val_seed=val_seed,
        dataset_root=dataset_root,
        splits_root=splits_root,
        supervision_setting=supervision_setting,
        weak_labels_path=weak_labels_path,
        weak_pos_ratio=weak_pos_ratio,
        weak_neg_ratio=weak_neg_ratio,
    )
    if mode.lower() == 'train':
        return DataLoader(vd, batch_size=1, shuffle=True)
    return vd


if __name__ == '__main__':
    pass
