# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

import numpy as np


def maybe_float(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def summarize_values(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    if not values:
        return {'mean': None, 'std': None, 'n': 0}
    values = np.asarray(values, dtype=float)
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        'n': int(len(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root', type=str, required=True)
    parser.add_argument('--video_type', type=str, default='SumMe')
    parser.add_argument('--n_splits', type=int, default=5)
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    rows = []
    rank_protocol = None
    for split_index in range(args.n_splits):
        result_dir = exp_root / args.video_type / 'results' / f'split{split_index}'
        metrics_path = result_dir / 'final_metrics.json'
        selection_path = exp_root / args.video_type / 'models' / f'split{split_index}' / 'selection.json'
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        with open(metrics_path) as f:
            metrics = json.loads(f.read())
        rank_metrics = metrics.get('rank_metrics') or {}
        if rank_protocol is None and isinstance(rank_metrics, dict):
            rank_protocol = rank_metrics.get('protocol')
        row = {
            'split': split_index,
            'fscore': maybe_float(metrics.get('fscore')),
            'kendall_tau': maybe_float(metrics.get('kendall_tau')),
            'spearman_rho': maybe_float(metrics.get('spearman_rho')),
            'rank_n_videos': rank_metrics.get('n_videos') if isinstance(rank_metrics, dict) else None,
            'rank_n_valid_kendall_tau': rank_metrics.get('n_valid_kendall_tau') if isinstance(rank_metrics, dict) else None,
            'rank_n_valid_spearman_rho': rank_metrics.get('n_valid_spearman_rho') if isinstance(rank_metrics, dict) else None,
            'selected_epoch': metrics.get('selected_epoch'),
        }
        if selection_path.exists():
            with open(selection_path) as f:
                selection = json.loads(f.read())
            row['selected_val_metrics'] = selection.get('selected_val_metrics')
            row['n_train'] = len(selection.get('train_keys', []))
            row['n_val'] = len(selection.get('val_keys', []))
            row['n_test'] = len(selection.get('test_keys', []))
        rows.append(row)

    summary = {
        'video_type': args.video_type,
        'exp_root': str(exp_root),
        'rank_protocol': rank_protocol,
        'splits': rows,
    }
    for key in ('fscore', 'kendall_tau', 'spearman_rho'):
        summary[key] = summarize_values(rows, key)

    output_path = exp_root / args.video_type / 'cv_summary.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
