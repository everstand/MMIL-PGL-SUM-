# -*- coding: utf-8 -*-
import json
import math
import subprocess
import sys
from pathlib import Path

from configs import get_config
from solver import Solver
from data_loader import get_loader


def compute_final_fscore(config, score_dir):
    """Run the existing evaluation script on the final clean-protocol score file."""
    eval_method = 'avg' if config.video_type.lower() == 'tvsum' else 'max'
    subprocess.run([
        sys.executable,
        'evaluation/compute_fscores.py',
        '--path', str(score_dir),
        '--dataset', config.video_type,
        '--eval', eval_method,
        '--dataset_root', str(config.dataset_root),
    ], check=True)
    with open(score_dir.joinpath('f_scores.txt')) as f:
        scores = json.loads(f.read())
    if len(scores) != 1:
        raise ValueError('Final test evaluation must contain exactly one score file.')
    return float(scores[0])


def clear_final_scores(score_dir):
    """Remove stale final-score files for deterministic clean-protocol reporting."""
    if not score_dir.exists():
        return
    for path in score_dir.glob('*.json'):
        path.unlink()
    for path in score_dir.glob('*.jsonl'):
        path.unlink()
    f_scores = score_dir.joinpath('f_scores.txt')
    if f_scores.exists():
        f_scores.unlink()


def loader_keys(loader):
    dataset = getattr(loader, 'dataset', loader)
    return list(getattr(dataset, 'dataset_keys', []))


def assert_disjoint(name_a, keys_a, name_b, keys_b):
    overlap = sorted(set(keys_a) & set(keys_b))
    if overlap:
        raise ValueError(f'{name_a} and {name_b} overlap: {overlap}')


def _mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(var)


def _format_metric_line(scope, name, mean, std):
    if mean is None or std is None:
        return f'{scope} {name} = N/A'
    return f'{scope} {name} = {mean:.4f} ± {std:.4f}'


def write_cv_summary(config):
    dataset_dir = Path(config.exp_root).joinpath(config.video_type)
    results_dir = dataset_dir.joinpath('results')
    models_dir = dataset_dir.joinpath('models')
    split_entries = []
    rank_protocol = None
    for final_metrics_path in sorted(results_dir.glob('split*/final_metrics.json')):
        split_name = final_metrics_path.parent.name
        split_idx = int(split_name.replace('split', ''))
        final_metrics = json.loads(final_metrics_path.read_text())
        entry = {
            'split': split_idx,
            'fscore': final_metrics.get('fscore', final_metrics.get('f_score')),
            'kendall_tau': final_metrics.get('kendall_tau'),
            'spearman_rho': final_metrics.get('spearman_rho'),
            'selected_epoch': final_metrics.get('selected_epoch'),
        }
        rank_metrics = final_metrics.get('rank_metrics', {})
        if rank_protocol is None and rank_metrics.get('protocol') is not None:
            rank_protocol = rank_metrics.get('protocol')
        if rank_metrics:
            entry['rank_n_videos'] = rank_metrics.get('n_videos')
            entry['rank_n_valid_kendall_tau'] = rank_metrics.get('n_valid_kendall_tau')
            entry['rank_n_valid_spearman_rho'] = rank_metrics.get('n_valid_spearman_rho')
        selection_path = models_dir.joinpath(split_name, 'selection.json')
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            if selection.get('selected_val_metrics') is not None:
                entry['selected_val_metrics'] = selection.get('selected_val_metrics')
            if selection.get('train_keys') is not None:
                entry['n_train'] = len(selection.get('train_keys', []))
            if selection.get('val_keys') is not None:
                entry['n_val'] = len(selection.get('val_keys', []))
            if selection.get('test_keys') is not None:
                entry['n_test'] = len(selection.get('test_keys', []))
        split_entries.append(entry)

    summary = {
        'video_type': config.video_type,
        'exp_root': str(config.exp_root),
        'completed_splits': len(split_entries),
        'expected_splits': 5,
        'is_complete': len(split_entries) == 5,
        'rank_protocol': rank_protocol,
        'splits': split_entries,
    }
    f1_values = [entry['fscore'] for entry in split_entries if entry.get('fscore') is not None]
    tau_values = [entry['kendall_tau'] for entry in split_entries if entry.get('kendall_tau') is not None]
    rho_values = [entry['spearman_rho'] for entry in split_entries if entry.get('spearman_rho') is not None]
    f1_mean, f1_std = _mean_std(f1_values)
    tau_mean, tau_std = _mean_std(tau_values)
    rho_mean, rho_std = _mean_std(rho_values)
    summary['aggregate'] = {
        'fscore_mean': f1_mean,
        'fscore_std': f1_std,
        'kendall_tau_mean': tau_mean,
        'kendall_tau_std': tau_std,
        'spearman_rho_mean': rho_mean,
        'spearman_rho_std': rho_std,
        'n_splits': len(split_entries),
    }
    summary['formatted_summary'] = {
        'scope': 'test',
        'lines': [
            _format_metric_line('test', 'F1', f1_mean, f1_std),
            _format_metric_line('test', 'Tau', tau_mean, tau_std),
            _format_metric_line('test', 'Rho', rho_mean, rho_std),
        ],
        'text': '\n'.join([
            _format_metric_line('test', 'F1', f1_mean, f1_std),
            _format_metric_line('test', 'Tau', tau_mean, tau_std),
            _format_metric_line('test', 'Rho', rho_mean, rho_std),
        ]),
        'note': 'This dataset-level cross-validation summary reports final test metrics only. Validation metrics remain split-level under selected_val_metrics.',
    }
    summary_path = dataset_dir.joinpath('cv_summary.json')
    summary_path.write_text(json.dumps(summary, indent=2))
    summary_path.chmod(0o777)


def update_final_summary(config, selected_epoch, final_metrics, test_loader):
    summary_path = config.save_dir.joinpath('selection.json')
    with open(summary_path) as f:
        summary = json.loads(f.read())
    summary['final_test_metrics'] = final_metrics
    summary['test_keys'] = loader_keys(test_loader)
    summary['test_evaluation'] = 'single final evaluation after inner-validation checkpoint selection'
    summary['selected_checkpoint'] = str(config.save_dir.joinpath(f'epoch-{selected_epoch}.pt'))
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    summary_path.chmod(0o777)
    write_cv_summary(config)


if __name__ == '__main__':
    """Main function that sets the data loaders; trains and evaluates the model."""
    config = get_config(mode='train')

    print(config)
    print('Currently selected split_index:', config.split_index)
    use_val_split = config.protocol == 'clean'
    train_loader = get_loader(
        config.mode, config.video_type, config.split_index,
        use_val_split=use_val_split,
        val_ratio=config.val_ratio, val_seed=config.val_seed,
        dataset_root=config.dataset_root,
        splits_root=config.splits_root,
        supervision_setting=config.supervision_setting,
        weak_labels_path=config.weak_labels_path,
        weak_pos_ratio=config.weak_pos_ratio,
        weak_neg_ratio=config.weak_neg_ratio,
    )
    val_loader = None
    test_loader = None
    if config.protocol == 'paper':
        test_config = get_config(mode='test')
        print(test_config)
        test_loader = get_loader(
            test_config.mode, test_config.video_type, test_config.split_index,
            dataset_root=config.dataset_root,
            splits_root=config.splits_root,
            supervision_setting=config.supervision_setting,
            weak_labels_path=config.weak_labels_path,
            weak_pos_ratio=config.weak_pos_ratio,
            weak_neg_ratio=config.weak_neg_ratio,
        )
    else:
        val_loader = get_loader(
            'val', config.video_type, config.split_index,
            use_val_split=True,
            val_ratio=config.val_ratio, val_seed=config.val_seed,
            dataset_root=config.dataset_root,
            splits_root=config.splits_root,
            supervision_setting=config.supervision_setting,
            weak_labels_path=config.weak_labels_path,
            weak_pos_ratio=config.weak_pos_ratio,
            weak_neg_ratio=config.weak_neg_ratio,
        )
        assert_disjoint('train_keys', loader_keys(train_loader), 'val_keys', loader_keys(val_loader))

    solver = Solver(config, train_loader, val_loader=val_loader, test_loader=test_loader)

    solver.build()
    if config.protocol == 'paper':
        solver.evaluate(-1)
        solver.train()
    else:
        selected_epoch = solver.train()
        print(f'Selected epoch: {selected_epoch} ({config.selection_metric})')
        test_config = get_config(mode='test')
        print(test_config)
        test_loader = get_loader(
            test_config.mode, test_config.video_type, test_config.split_index,
            use_val_split=True,
            val_ratio=config.val_ratio, val_seed=config.val_seed,
            dataset_root=config.dataset_root,
            splits_root=config.splits_root,
            supervision_setting=config.supervision_setting,
            weak_labels_path=config.weak_labels_path,
            weak_pos_ratio=config.weak_pos_ratio,
            weak_neg_ratio=config.weak_neg_ratio,
        )
        assert_disjoint('train_keys', loader_keys(train_loader), 'test_keys', loader_keys(test_loader))
        assert_disjoint('val_keys', loader_keys(val_loader), 'test_keys', loader_keys(test_loader))
        solver.test_loader = test_loader
        solver.load_checkpoint(selected_epoch)
        final_score_dir = config.score_dir.joinpath('final')
        clear_final_scores(final_score_dir)
        scores_path = solver.evaluate(selected_epoch, score_dir=final_score_dir)
        official_fscore = compute_final_fscore(config, final_score_dir)
        final_metrics = solver.compute_metrics(scores_path)
        final_metrics['fscore'] = official_fscore
        final_metrics['selected_epoch'] = int(selected_epoch)
        with open(config.score_dir.joinpath('final_metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=2)
        update_final_summary(config, selected_epoch, final_metrics, test_loader)
        print(f'Final test metrics: {final_metrics}')
