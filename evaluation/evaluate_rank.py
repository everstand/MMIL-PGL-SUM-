# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

from rank_metrics import compute_rank_metrics


def score_files_from_args(scores_path=None, score_dir=None, dataset=None):
    if scores_path:
        return [Path(scores_path)]
    if not score_dir:
        raise ValueError('Either --scores_path or --score_dir is required')
    score_dir = Path(score_dir)
    dataset_prefix = dataset.lower() if dataset else ''
    files = sorted(
        path for path in score_dir.glob('*.json')
        if path.stem.lower().startswith(dataset_prefix + '_')
    )
    if not files:
        raise FileNotFoundError(f'No score json files found under {score_dir}')
    return files


def main():
    parser = argparse.ArgumentParser(description='Compute strict SumMe/TVSum rank-correlation metrics.')
    parser.add_argument('--dataset', choices=['SumMe', 'TVSum', 'summe', 'tvsum'], required=True)
    parser.add_argument('--scores_path', type=str, default=None,
                        help='Path to one model score json file')
    parser.add_argument('--score_dir', type=str, default=None,
                        help='Directory containing model score json files')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Optional dataset h5 path')
    parser.add_argument('--tvsum_anno_path', type=str, default=None,
                        help='Path to official TVSum ydata-anno.tsv annotations')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional output json path')
    args = parser.parse_args()

    results = []
    for score_path in score_files_from_args(args.scores_path, args.score_dir, args.dataset):
        metrics = compute_rank_metrics(
            score_path,
            args.dataset,
            dataset_path=args.dataset_path,
            tvsum_anno_path=args.tvsum_anno_path,
        )
        metrics['scores_path'] = str(score_path)
        results.append(metrics)

    if len(results) == 1:
        output = results[0]
    else:
        output = {'dataset': args.dataset, 'score_files': results}

    text = json.dumps(output, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
