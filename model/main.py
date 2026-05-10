# -*- coding: utf-8 -*-
import json
import subprocess
import sys

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


if __name__ == '__main__':
    """Main function that sets the data loaders; trains and evaluates the model."""
    config = get_config(mode='train')

    print(config)
    print('Currently selected split_index:', config.split_index)
    use_val_split = config.protocol == 'clean'
    train_loader = get_loader(config.mode, config.video_type, config.split_index,
                              use_val_split=use_val_split,
                              val_ratio=config.val_ratio, val_seed=config.val_seed,
                              dataset_root=config.dataset_root,
                              splits_root=config.splits_root)
    val_loader = None
    test_loader = None
    if config.protocol == 'paper':
        test_config = get_config(mode='test')
        print(test_config)
        test_loader = get_loader(test_config.mode, test_config.video_type, test_config.split_index,
                                 dataset_root=config.dataset_root,
                                 splits_root=config.splits_root)
    else:
        val_loader = get_loader('val', config.video_type, config.split_index,
                                use_val_split=True,
                                val_ratio=config.val_ratio, val_seed=config.val_seed,
                                dataset_root=config.dataset_root,
                                splits_root=config.splits_root)
        assert_disjoint('train_keys', loader_keys(train_loader), 'val_keys', loader_keys(val_loader))

    solver = Solver(config, train_loader, val_loader=val_loader, test_loader=test_loader)

    solver.build()
    if config.protocol == 'paper':
        solver.evaluate(-1)  # evaluates summaries using the initial random weights
        solver.train()
    else:
        selected_epoch = solver.train()
        print(f'Selected epoch: {selected_epoch} ({config.selection_metric})')
        test_config = get_config(mode='test')
        print(test_config)
        test_loader = get_loader(test_config.mode, test_config.video_type, test_config.split_index,
                                 use_val_split=True,
                                 val_ratio=config.val_ratio, val_seed=config.val_seed,
                                 dataset_root=config.dataset_root,
                                 splits_root=config.splits_root)
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
# tensorboard --logdir '../PGL-SUM/experiments/results/'
