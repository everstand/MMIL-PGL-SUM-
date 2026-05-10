# -*- coding: utf-8 -*-
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / 'evaluation'
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
from rank_metrics import compute_rank_metrics


@dataclass
class Job:
    dataset: str
    split: int
    gpu: int
    batch_size: int


class JobMonitor:
    def __init__(self, job, args):
        self.job = job
        self.args = args
        self.exp_root = Path(args.exp_root)
        self.prefix = f'[{job.dataset} split{job.split} gpu{job.gpu}]'
        self.log_dir = self.exp_root / job.dataset / 'logs' / f'split{job.split}'
        self.score_dir = self.exp_root / job.dataset / 'results' / f'split{job.split}'
        self.val_dir = self.score_dir / 'val'
        self.model_dir = self.exp_root / job.dataset / 'models' / f'split{job.split}'
        self.stdout_log = Path(args.log_root) / f'{job.dataset}_split{job.split}.stdout.log'
        self.seen_epochs = set()
        self.best_val = None
        self.best_epoch = None
        self.start_time = None

    def command(self):
        return [
            sys.executable, 'tools/train_with_fast_flush.py',
            '--video_type', self.job.dataset,
            '--split_index', str(self.job.split),
            '--protocol', 'clean',
            '--exp_root', str(self.exp_root),
            '--save_checkpoints', 'true',
            '--selection_metric', 'val_fscore',
            '--seed', str(self.args.seed),
            '--val_seed', str(self.args.val_seed),
            '--val_ratio', str(self.args.val_ratio),
            '--n_epochs', str(self.args.n_epochs),
            '--batch_size', str(self.job.batch_size),
            '--n_segments', str(self.args.n_segments),
            '--heads', str(self.args.heads),
            '--fusion', self.args.fusion,
            '--pos_enc', self.args.pos_enc,
            '--lr', str(self.args.lr),
            '--l2_req', str(self.args.l2_req),
            '--early_stop_patience', str(self.args.early_stop_patience),
            '--tvsum_anno_path', str(REPO_ROOT / 'data/datasets/TVSum/ydata-anno.tsv'),
        ]

    def print_header(self):
        msg = {
            'dataset': self.job.dataset,
            'split': self.job.split,
            'gpu': self.job.gpu,
            'seed': self.args.seed,
            'val_seed': self.args.val_seed,
            'val_ratio': self.args.val_ratio,
            'n_epochs': self.args.n_epochs,
            'batch_size': self.job.batch_size,
            'model': {
                'n_segments': self.args.n_segments,
                'heads': self.args.heads,
                'fusion': self.args.fusion,
                'pos_enc': self.args.pos_enc,
            },
            'optim': {'lr': self.args.lr, 'l2_req': self.args.l2_req},
            'exp_root': str(self.exp_root),
        }
        print(f'{self.prefix} CONFIG {json.dumps(msg, sort_keys=True)}', flush=True)

    def read_losses(self):
        losses = {}
        if not self.log_dir.exists():
            return losses
        for event_file in self.log_dir.glob('events.out.tfevents.*'):
            try:
                acc = EventAccumulator(str(event_file), size_guidance={'scalars': 0})
                acc.Reload()
                if 'loss_epoch' not in acc.Tags().get('scalars', []):
                    continue
                for item in acc.Scalars('loss_epoch'):
                    losses[int(item.step)] = float(item.value)
            except Exception:
                continue
        return losses

    def read_val_metrics(self):
        metrics_path = self.val_dir / 'metrics.jsonl'
        rows = {}
        if not metrics_path.exists():
            return rows
        with metrics_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'epoch' in row:
                    rows[int(row['epoch'])] = row
        return rows

    def score_file(self, epoch):
        return self.val_dir / f'{self.job.dataset}_{epoch}.json'

    def emit_new_epochs(self, force=False):
        losses = self.read_losses()
        val_rows = self.read_val_metrics()
        for epoch in sorted(val_rows):
            if epoch in self.seen_epochs:
                continue
            score_path = self.score_file(epoch)
            if not score_path.exists():
                continue
            loss = losses.get(epoch)
            if loss is None and not force:
                continue
            row = val_rows[epoch]
            try:
                rank = compute_rank_metrics(score_path, self.job.dataset)
                tau = rank['kendall_tau']
                rho = rank['spearman_rho']
                n_videos = max(1, int(rank.get('n_videos') or 1))
                valid_tau = int(rank.get('n_valid_kendall_tau') or 0)
                valid_rho = int(rank.get('n_valid_spearman_rho') or 0)
                val_cov = min(valid_tau, valid_rho) / n_videos
            except Exception as exc:
                tau = None
                rho = None
                val_cov = 0.0
                print(f'{self.prefix} rank_metric_error epoch={epoch:03d}: {exc}', flush=True)
            val_f1 = float(row.get('fscore')) / 100.0
            if self.best_val is None or val_f1 > self.best_val:
                self.best_val = val_f1
                self.best_epoch = epoch
            loss_text = 'NA' if loss is None else f'{loss:.4f}'
            tau_text = 'NA' if tau is None else f'{tau:.4f}'
            rho_text = 'NA' if rho is None else f'{rho:.4f}'
            total_epochs = int(self.args.n_epochs)
            print(
                f'{self.job.dataset} split{self.job.split} gpu{self.job.gpu} | '
                f'Epoch {epoch + 1:03d}/{total_epochs:03d} | '
                f'loss={loss_text} | '
                f'val_F1={val_f1:.4f} | val_Tau={tau_text} | val_Rho={rho_text} | '
                f'val_cov={val_cov:.4f} | best_val_F1={self.best_val:.4f}',
                flush=True,
            )
            self.seen_epochs.add(epoch)

    def run(self):
        self.print_header()
        self.stdout_log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.job.gpu)
        env.pop('LD_LIBRARY_PATH', None)
        self.start_time = time.time()
        proc = subprocess.Popen(
            self.command(),
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        output_queue = queue.Queue()

        def reader():
            with self.stdout_log.open('w') as log_file:
                for line in proc.stdout:
                    log_file.write(line)
                    log_file.flush()
                    output_queue.put(line.rstrip('\n'))

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        while proc.poll() is None:
            self.emit_new_epochs()
            try:
                while True:
                    line = output_queue.get_nowait()
                    if 'Selected epoch:' in line:
                        print(f'{self.prefix} {line}', flush=True)
            except queue.Empty:
                pass
            time.sleep(self.args.poll_interval)
        thread.join(timeout=5)
        self.emit_new_epochs()
        code = proc.returncode
        if code != 0:
            print(f'{self.prefix} FAILED returncode={code} log={self.stdout_log}', flush=True)
            return code
        final_path = self.score_dir / 'final_metrics.json'
        if final_path.exists():
            metrics = json.loads(final_path.read_text())
            print(
                f'{self.prefix} FINAL selected_epoch={metrics.get("selected_epoch")} '
                f'test_F1={float(metrics["fscore"]):.4f} '
                f'test_tau={float(metrics["kendall_tau"]):.6f} '
                f'test_rho={float(metrics["spearman_rho"]):.6f}',
                flush=True,
            )
        else:
            print(f'{self.prefix} WARNING missing final_metrics.json', flush=True)
            return 1
        return 0


def summarize_dataset(exp_root, dataset, n_splits):
    cmd = [
        sys.executable, 'model/summarize_cv.py',
        '--exp_root', str(exp_root),
        '--video_type', dataset,
        '--n_splits', str(n_splits),
    ]
    output = subprocess.check_output(cmd, cwd=str(REPO_ROOT), text=True)
    summary = json.loads(output)
    print(
        f'[SUMMARY {dataset}] F1={summary["fscore"]["mean"]:.4f}±{summary["fscore"]["std"]:.4f} '
        f'tau={summary["kendall_tau"]["mean"]:.6f}±{summary["kendall_tau"]["std"]:.6f} '
        f'rho={summary["spearman_rho"]["mean"]:.6f}±{summary["spearman_rho"]["std"]:.6f} '
        f'n={summary["fscore"]["n"]}',
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['SumMe', 'TVSum'])
    parser.add_argument('--splits', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--gpus', nargs='+', type=int, default=[6, 7])
    parser.add_argument('--exp_root', type=str, required=True)
    parser.add_argument('--log_root', type=str, required=True)
    parser.add_argument('--seed', type=int, default=9999)
    parser.add_argument('--val_seed', type=int, default=0)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--summe_batch_size', type=int, default=20)
    parser.add_argument('--tvsum_batch_size', type=int, default=40)
    parser.add_argument('--n_segments', type=int, default=4)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--fusion', type=str, default='add')
    parser.add_argument('--pos_enc', type=str, default='absolute')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--l2_req', type=float, default=1e-5)
    parser.add_argument('--early_stop_patience', type=int, default=0)
    parser.add_argument('--poll_interval', type=float, default=2.0)
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    log_root = Path(args.log_root)
    exp_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for dataset in args.datasets:
        batch_size = args.summe_batch_size if dataset.lower() == 'summe' else args.tvsum_batch_size
        for split in args.splits:
            jobs.append((dataset, split, batch_size))

    print(f'[RUN] exp_root={exp_root} log_root={log_root} jobs={len(jobs)} gpus={args.gpus}', flush=True)
    failures = []
    for idx in range(0, len(jobs), len(args.gpus)):
        batch = []
        for gpu, item in zip(args.gpus, jobs[idx:idx + len(args.gpus)]):
            dataset, split, batch_size = item
            monitor = JobMonitor(Job(dataset, split, gpu, batch_size), args)
            thread = threading.Thread(target=lambda m=monitor: batch.append((m, m.run())))
            thread.start()
            batch.append((thread, None))
        # Wait for monitor threads. The list contains placeholders plus completed records.
        for thread, _ in list(batch):
            if isinstance(thread, threading.Thread):
                thread.join()
        for item in batch:
            if isinstance(item[0], JobMonitor) and item[1] != 0:
                failures.append((item[0].job, item[1]))
        if failures:
            break

    if failures:
        print(f'[RUN] failures={failures}', flush=True)
        raise SystemExit(1)

    for dataset in args.datasets:
        summarize_dataset(exp_root, dataset, len(args.splits))


if __name__ == '__main__':
    main()
