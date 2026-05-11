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


def _normalize_shot_target(shot_target, shot_mask, norm_mode='per_video_minmax'):
    shot_target = np.asarray(shot_target, dtype=np.float32).copy()
    shot_mask = np.asarray(shot_mask, dtype=bool)

    if norm_mode == 'none':
        return shot_target

    valid = np.flatnonzero(shot_mask)
    if valid.size == 0:
        raise ValueError('shot_mask has no valid entries.')

    vals = shot_target[valid]
    vmin, vmax = float(vals.min()), float(vals.max())

    if vmax > vmin:
        shot_target[valid] = (vals - vmin) / (vmax - vmin)
    else:
        shot_target[valid] = 0.5

    shot_target[valid] = np.clip(shot_target[valid], 0.0, 1.0)
    return shot_target


def _expand_shot_utility_to_steps(video_name, video_payload, hdf, norm_mode='per_video_minmax'):
    if not isinstance(video_payload, dict):
        raise TypeError(f'Weak label payload for {video_name} must be a dict.')
    if 'shot_utility' not in video_payload:
        raise KeyError(f'Missing shot_utility for {video_name}')

    shot_target = np.asarray(video_payload['shot_utility'], dtype=np.float32)
    shot_mask = np.asarray(
        video_payload.get('event_valid_mask', np.ones_like(shot_target, dtype=bool)),
        dtype=bool,
    )

    change_points = np.asarray(hdf[f'{video_name}/change_points'])
    picks = np.asarray(hdf[f'{video_name}/picks'])
    frame_features = np.asarray(hdf[f'{video_name}/features'])

    if shot_target.ndim != 1:
        raise ValueError(f'{video_name}: shot_target must be 1D.')
    if shot_mask.ndim != 1:
        raise ValueError(f'{video_name}: shot_mask must be 1D.')
    if len(shot_target) != len(change_points):
        raise ValueError(f'{video_name}: shot_target length mismatch.')
    if len(shot_mask) != len(change_points):
        raise ValueError(f'{video_name}: shot_mask length mismatch.')

    _validate_finite(f'{video_name} shot_target', shot_target)
    shot_target = _normalize_shot_target(shot_target, shot_mask, norm_mode=norm_mode)

    shot_end = change_points[:, 1]
    step_shot_idx = np.searchsorted(shot_end, picks, side='left')
    step_shot_idx = np.clip(step_shot_idx, 0, len(shot_target) - 1)

    starts = change_points[step_shot_idx, 0]
    ends = change_points[step_shot_idx, 1]
    if np.any((picks < starts) | (picks > ends)):
        raise ValueError(f'{video_name}: failed shot-step alignment.')

    step_target = shot_target[step_shot_idx].astype(np.float32)
    step_mask = shot_mask[step_shot_idx].astype(bool)

    if len(step_target) != len(frame_features):
        raise ValueError(f'{video_name}: step_target length mismatch.')
    if int(step_mask.sum()) <= 0:
        raise ValueError(f'{video_name}: step_mask.sum() must be > 0.')

    shot_len_steps = np.bincount(step_shot_idx, minlength=len(shot_target)).astype(np.int64)
    _validate_finite(f'{video_name} step_target', step_target)
    valid_step_values = step_target[step_mask]
    if valid_step_values.size == 0:
        raise ValueError(f'{video_name}: step_target has no valid entries.')
    step_min = float(valid_step_values.min())
    step_max = float(valid_step_values.max())
    if step_min < -1e-6 or step_max > 1.0 + 1e-6:
        raise ValueError(f'{video_name}: normalized step_target is outside [0, 1].')

    return step_target, step_mask, shot_target, shot_mask, step_shot_idx.astype(np.int64), shot_len_steps


def _build_supervised_step_targets(video_name, hdf):
    step_target = np.asarray(hdf[f'{video_name}/gtscore'], dtype=np.float32)
    change_points = np.asarray(hdf[f'{video_name}/change_points'])
    picks = np.asarray(hdf[f'{video_name}/picks'])
    frame_features = np.asarray(hdf[f'{video_name}/features'])

    if step_target.ndim != 1:
        raise ValueError(f'{video_name}: gtscore must be 1D.')
    if len(step_target) != len(frame_features):
        raise ValueError(f'{video_name}: gtscore length mismatch.')
    _validate_finite(f'{video_name} gtscore', step_target)

    shot_end = change_points[:, 1]
    step_shot_idx = np.searchsorted(shot_end, picks, side='left')
    step_shot_idx = np.clip(step_shot_idx, 0, len(change_points) - 1)
    starts = change_points[step_shot_idx, 0]
    ends = change_points[step_shot_idx, 1]
    if np.any((picks < starts) | (picks > ends)):
        raise ValueError(f'{video_name}: failed shot-step alignment.')

    shot_len_steps = np.bincount(step_shot_idx, minlength=len(change_points)).astype(np.int64)
    shot_sum = np.bincount(step_shot_idx, weights=step_target, minlength=len(change_points)).astype(np.float32)
    shot_target = np.zeros(len(change_points), dtype=np.float32)
    shot_mask = shot_len_steps > 0
    shot_target[shot_mask] = shot_sum[shot_mask] / shot_len_steps[shot_mask].astype(np.float32)

    return (
        step_target.astype(np.float32),
        np.ones_like(step_target, dtype=bool),
        shot_target,
        shot_mask.astype(bool),
        step_shot_idx.astype(np.int64),
        shot_len_steps,
    )


class VideoData(Dataset):
    def __init__(self, mode, video_type, split_index, use_val_split=False,
                 val_ratio=0.2, val_seed=0, dataset_root=None, splits_root=None,
                 supervision_setting='supervised', weak_labels_path=None,
                 weak_pos_ratio=0.15, weak_neg_ratio=0.15,
                 weak_target_norm='per_video_minmax'):
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
        self.weak_target_norm = weak_target_norm

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
                    step_target_np, step_mask_np, shot_target_np, shot_mask_np, step_shot_idx_np, shot_len_steps_np = \
                        _expand_shot_utility_to_steps(
                            video_name, weak_label_data[video_name], hdf,
                            norm_mode=self.weak_target_norm,
                        )
                else:
                    step_target_np, step_mask_np, shot_target_np, shot_mask_np, step_shot_idx_np, shot_len_steps_np = \
                        _build_supervised_step_targets(video_name, hdf)

                self.train_samples.append({
                    'frame_features': frame_features,
                    'step_target': torch.tensor(step_target_np, dtype=torch.float32),
                    'step_mask': torch.tensor(step_mask_np, dtype=torch.bool),
                    'shot_target': torch.tensor(shot_target_np, dtype=torch.float32),
                    'shot_mask': torch.tensor(shot_mask_np, dtype=torch.bool),
                    'step_shot_idx': torch.tensor(step_shot_idx_np, dtype=torch.long),
                    'shot_len_steps': torch.tensor(shot_len_steps_np, dtype=torch.long),
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


def _single_item_collate(batch):
    if len(batch) != 1:
        raise ValueError(f'Expected batch_size=1 for train loader, got {len(batch)} items.')
    return batch[0]


def get_loader(mode, video_type, split_index, use_val_split=False,
               val_ratio=0.2, val_seed=0, dataset_root=None, splits_root=None,
               supervision_setting='supervised', weak_labels_path=None,
               weak_pos_ratio=0.15, weak_neg_ratio=0.15,
               weak_target_norm='per_video_minmax'):
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
        weak_target_norm=weak_target_norm,
    )
    if mode.lower() == 'train':
        return DataLoader(vd, batch_size=1, shuffle=True, collate_fn=_single_item_collate)
    return vd


if __name__ == '__main__':
    pass

