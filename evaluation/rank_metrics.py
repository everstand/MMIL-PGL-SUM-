# -*- coding: utf-8 -*-
"""Rank-correlation evaluation for video summarization.

This module keeps the rank metric protocol inside the repository instead of
calling external evaluation code.
"""
import csv
import json
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr


DATASET_FILENAMES = {
    'summe': 'eccv16_dataset_summe_google_pool5.h5',
    'tvsum': 'eccv16_dataset_tvsum_google_pool5.h5',
}


def repo_root():
    return Path(__file__).resolve().parents[1]


def default_dataset_path(dataset):
    dataset_key = dataset.lower()
    if dataset_key not in DATASET_FILENAMES:
        raise ValueError(f'Unsupported dataset: {dataset}')
    dataset_dir = 'SumMe' if dataset_key == 'summe' else 'TVSum'
    return repo_root().joinpath('data', 'datasets', dataset_dir, DATASET_FILENAMES[dataset_key])


def default_tvsum_anno_path():
    return repo_root().joinpath('data', 'datasets', 'TVSum', 'ydata-anno.tsv')


def load_scores(scores_path):
    with open(scores_path) as f:
        raw_scores = json.load(f)
    return {key: np.asarray(value, dtype=np.float64).reshape(-1) for key, value in raw_scores.items()}


def video_index(video_key):
    try:
        return int(str(video_key).split('_')[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f'Expected video key like video_1, got {video_key}') from exc


def clean_scalar(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def mean_optional(values):
    valid_values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not valid_values:
        return None
    return float(np.mean(valid_values))


def std_optional(values):
    valid_values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if len(valid_values) < 2:
        return 0.0 if valid_values else None
    return float(np.std(valid_values, ddof=1))


def sample_at_picks(frame_scores, picks, context):
    frame_scores = np.asarray(frame_scores, dtype=np.float64).reshape(-1)
    picks = np.asarray(picks, dtype=np.int64).reshape(-1)
    if np.any(picks < 0) or np.any(picks >= len(frame_scores)):
        raise ValueError(f'{context}: picks are outside the reference score length')
    return frame_scores[picks]


def align_pair(pred_scores, ref_scores, context):
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    ref_scores = np.asarray(ref_scores, dtype=np.float64).reshape(-1)
    if len(pred_scores) == len(ref_scores):
        return pred_scores, ref_scores
    if abs(len(pred_scores) - len(ref_scores)) <= 1:
        length = min(len(pred_scores), len(ref_scores))
        return pred_scores[:length], ref_scores[:length]
    raise ValueError(
        f'{context}: prediction/reference length mismatch '
        f'({len(pred_scores)} vs {len(ref_scores)})'
    )


def scipy_rank_correlations(pred_scores, ref_scores, context):
    pred_scores, ref_scores = align_pair(pred_scores, ref_scores, context)
    if len(pred_scores) < 2:
        return None, None, 'too_short'
    if np.isnan(pred_scores).any() or np.isnan(ref_scores).any():
        raise ValueError(f'{context}: NaN found before rank correlation')
    if np.unique(pred_scores).size < 2 or np.unique(ref_scores).size < 2:
        return None, None, 'constant_sequence'

    rho = spearmanr(pred_scores, ref_scores, nan_policy='raise').correlation
    try:
        tau = kendalltau(
            rankdata(pred_scores),
            rankdata(ref_scores),
            variant='b',
            nan_policy='raise',
        ).correlation
    except TypeError as exc:
        if 'variant' not in str(exc):
            raise
        # Older SciPy exposes tau-b as the default but has no variant keyword.
        tau = kendalltau(
            rankdata(pred_scores),
            rankdata(ref_scores),
            nan_policy='raise',
        ).correlation
    return clean_scalar(tau), clean_scalar(rho), None


def parse_tvsum_annotation_row(row, line_number, anno_path):
    if len(row) < 3:
        raise ValueError(f'{anno_path}:{line_number}: expected at least 3 TSV columns')
    video_name = row[0]
    try:
        scores = np.asarray([float(value) for value in row[2].split(',')], dtype=np.float64)
    except ValueError as exc:
        if line_number == 1:
            return None, None
        raise ValueError(f'{anno_path}:{line_number}: invalid annotation scores') from exc
    if len(scores) == 0:
        raise ValueError(f'{anno_path}:{line_number}: empty annotation scores')
    max_score = np.max(scores)
    if max_score > 0:
        scores = scores / max_score
    return video_name, scores


def load_tvsum_annotations(anno_path=None):
    anno_path = Path(anno_path) if anno_path else default_tvsum_anno_path()
    if not anno_path.exists():
        raise FileNotFoundError(
            f'TVSum rank metrics require the official ydata annotations: {anno_path}'
        )

    grouped_annotations = OrderedDict()
    with open(anno_path, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            video_name, scores = parse_tvsum_annotation_row(row, line_number, anno_path)
            if video_name is None:
                continue
            grouped_annotations.setdefault(video_name, []).append(scores)

    if not grouped_annotations:
        raise ValueError(f'No TVSum annotations found in {anno_path}')
    return list(grouped_annotations.items()), str(anno_path)


def summe_video_rank(video_key, pred_scores, hdf_group):
    user_summary = np.asarray(hdf_group['user_summary'], dtype=np.float64)
    picks = np.asarray(hdf_group['picks'], dtype=np.int64)
    reference_frame_scores = np.mean(user_summary, axis=0)
    reference_scores = sample_at_picks(reference_frame_scores, picks, video_key)
    tau, rho, status = scipy_rank_correlations(pred_scores, reference_scores, video_key)
    return {
        'video': video_key,
        'kendall_tau': tau,
        'spearman_rho': rho,
        'status': status or 'ok',
        'aligned_length': int(min(len(pred_scores), len(reference_scores))),
        'reference': 'mean_user_summary_sampled_at_picks',
    }


def tvsum_video_rank(video_key, pred_scores, annotations_by_order):
    idx = video_index(video_key) - 1
    if idx < 0 or idx >= len(annotations_by_order):
        raise ValueError(f'{video_key}: no matching TVSum annotation group')
    source_video_name, annotator_scores = annotations_by_order[idx]

    annotator_rows = []
    for annotator_index, frame_scores in enumerate(annotator_scores):
        reference_scores = np.asarray(frame_scores, dtype=np.float64).reshape(-1)[::15]
        tau, rho, status = scipy_rank_correlations(
            pred_scores,
            reference_scores,
            f'{video_key}/annotator_{annotator_index}',
        )
        annotator_rows.append({
            'annotator': annotator_index,
            'kendall_tau': tau,
            'spearman_rho': rho,
            'status': status or 'ok',
            'aligned_length': int(min(len(pred_scores), len(reference_scores))),
        })

    return {
        'video': video_key,
        'source_video_name': source_video_name,
        'kendall_tau': mean_optional(row['kendall_tau'] for row in annotator_rows),
        'spearman_rho': mean_optional(row['spearman_rho'] for row in annotator_rows),
        'n_annotators': len(annotator_rows),
        'n_valid_kendall_tau': sum(row['kendall_tau'] is not None for row in annotator_rows),
        'n_valid_spearman_rho': sum(row['spearman_rho'] is not None for row in annotator_rows),
        'status': 'ok',
        'reference': 'per_annotator_ydata_scores_sampled_every_15_frames',
        'annotators': annotator_rows,
    }


def compute_rank_metrics(scores_path, dataset, dataset_path=None, tvsum_anno_path=None):
    dataset_key = dataset.lower()
    if dataset_key not in ('summe', 'tvsum'):
        raise ValueError(f'Unsupported dataset: {dataset}')

    scores_by_video = load_scores(scores_path)
    dataset_path = Path(dataset_path) if dataset_path else default_dataset_path(dataset)
    annotations_by_order = None
    annotation_path = None
    if dataset_key == 'tvsum':
        annotations_by_order, annotation_path = load_tvsum_annotations(tvsum_anno_path)

    videos = []
    with h5py.File(dataset_path, 'r') as hdf:
        for video_key in scores_by_video:
            if video_key not in hdf:
                raise KeyError(f'{video_key} not found in {dataset_path}')
            if dataset_key == 'summe':
                videos.append(summe_video_rank(video_key, scores_by_video[video_key], hdf[video_key]))
            else:
                videos.append(tvsum_video_rank(video_key, scores_by_video[video_key], annotations_by_order))

    taus = [row['kendall_tau'] for row in videos]
    rhos = [row['spearman_rho'] for row in videos]
    protocol = {
        'dataset': 'SumMe' if dataset_key == 'summe' else 'TVSum',
        'kendall': 'scipy.stats.kendalltau(rankdata(pred), rankdata(ref), variant="b", nan_policy="raise"); old SciPy fallback uses default tau-b',
        'spearman': 'scipy.stats.spearmanr(pred, ref, nan_policy="raise")',
        'nan_handling': 'constant or too-short sequences are recorded as null and excluded from means',
    }
    if dataset_key == 'summe':
        protocol.update({
            'reference': 'mean user_summary over annotators, then sample at h5 picks',
            'aggregation': 'video mean over test videos',
        })
    else:
        protocol.update({
            'reference': 'official TVSum ydata annotations, normalized per annotator and sampled every 15 frames',
            'aggregation': 'annotator mean within each video, then video mean over test videos',
            'annotation_path': annotation_path,
        })

    return {
        'kendall_tau': mean_optional(taus),
        'spearman_rho': mean_optional(rhos),
        'kendall_tau_std': std_optional(taus),
        'spearman_rho_std': std_optional(rhos),
        'n_videos': len(videos),
        'n_valid_kendall_tau': sum(value is not None for value in taus),
        'n_valid_spearman_rho': sum(value is not None for value in rhos),
        'protocol': protocol,
        'videos': videos,
    }
