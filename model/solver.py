# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import random
import json
import h5py
import sys
from pathlib import Path
from tqdm import tqdm, trange

EVAL_DIR = Path(__file__).resolve().parents[1].joinpath('evaluation')
if str(EVAL_DIR) not in sys.path:
    sys.path.append(str(EVAL_DIR))
from evaluation_metrics import evaluate_summary
from generate_summary import generate_summary
from rank_metrics import compute_rank_metrics

from layers.summarizer import PGL_SUM
from utils import TensorboardWriter


class Solver(object):
    def __init__(self, config=None, train_loader=None, val_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates PGL-SUM model"""
        # Initialize variables to None, to be safe
        self.model, self.optimizer, self.writer = None, None, None

        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_losses = []
        self.val_metrics = []

        # Set the seed for generating reproducible random numbers
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

    def build(self):
        """ Function for constructing the PGL-SUM model of its key modules and parameters."""
        # Model creation
        self.model = PGL_SUM(input_size=self.config.input_size,
                             output_size=self.config.input_size,
                             num_segments=self.config.n_segments,
                             heads=self.config.heads,
                             fusion=self.config.fusion,
                             pos_enc=self.config.pos_enc).to(self.config.device)
        if self.config.init_type is not None:
            self.init_weights(self.model, init_type=self.config.init_type, init_gain=self.config.init_gain)

        if self.config.mode == 'train':
            # Optimizer initialization
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_req)
            self.writer = TensorboardWriter(str(self.config.log_dir))

    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        """ Initialize 'net' network weights, based on the chosen 'init_type' and 'init_gain'.

        :param nn.Module net: Network to be initialized.
        :param str init_type: Name of initialization method: normal | xavier | kaiming | orthogonal.
        :param float init_gain: Scaling factor for normal.
        """
        for name, param in net.named_parameters():
            if 'weight' in name and "norm" not in name:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))  # ReLU activation function
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=np.sqrt(2.0))      # ReLU activation function
                else:
                    raise NotImplementedError(f"initialization method {init_type} is not implemented.")
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    criterion = nn.MSELoss()

    def compute_train_loss(self, output, target, weak_mask=None, pos_mask=None, neg_mask=None):
        """Compute train loss under the selected supervision paradigm."""
        output = output.view(-1)
        target = target.view(-1)
        if self.config.supervision_setting == 'supervised':
            loss = self.criterion(output, target)
            return loss, {'loss_mse': float(loss.detach().cpu().item())}

        weak_mask = weak_mask.view(-1).bool()
        pos_mask = pos_mask.view(-1).bool()
        neg_mask = neg_mask.view(-1).bool()
        if int(weak_mask.sum().item()) <= 0:
            raise ValueError('weak_mask.sum() must be > 0 for weak supervision.')

        weak_bce = F.binary_cross_entropy(output[weak_mask], target[weak_mask])
        if pos_mask.any() and neg_mask.any():
            rank_margin = self.config.weak_rank_margin - output[pos_mask].unsqueeze(1) + output[neg_mask].unsqueeze(0)
            weak_rank = torch.relu(rank_margin).mean()
        else:
            weak_rank = output.new_tensor(0.0)
        loss = weak_bce + self.config.weak_rank_weight * weak_rank
        return loss, {
            'loss_bce': float(weak_bce.detach().cpu().item()),
            'loss_rank': float(weak_rank.detach().cpu().item()),
        }

    def train(self):
        """ Main function to train the PGL-SUM model. """
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            self.model.train()

            loss_history = []
            component_history = {}
            if len(self.train_loader) == 0:
                raise ValueError('The training set is empty.')
            batch_size = min(self.config.batch_size, len(self.train_loader))
            num_batches = int(len(self.train_loader) / batch_size)  # full-batch or mini batch
            iterator = iter(self.train_loader)
            for _ in trange(num_batches, desc='Batch', ncols=80, leave=False):
                # ---- Training ... ----#
                if self.config.verbose:
                    tqdm.write('Time to train the model...')

                self.optimizer.zero_grad()
                for _ in trange(batch_size, desc='Video', ncols=80, leave=False):
                    batch = next(iterator)
                    frame_features = batch['frame_features'].to(self.config.device)
                    target = batch['target'].to(self.config.device)
                    weak_mask = batch['mask'].to(self.config.device)
                    pos_mask = batch['pos_mask'].to(self.config.device)
                    neg_mask = batch['neg_mask'].to(self.config.device)

                    output, weights = self.model(frame_features.squeeze(0))
                    loss, loss_components = self.compute_train_loss(
                        output.squeeze(0),
                        target.squeeze(0),
                        weak_mask=weak_mask.squeeze(0),
                        pos_mask=pos_mask.squeeze(0),
                        neg_mask=neg_mask.squeeze(0),
                    )

                    if self.config.verbose:
                        tqdm.write(f'[{epoch_i}] loss: {loss.item()}')

                    loss.backward()
                    loss_history.append(loss.detach())
                    for name, value in loss_components.items():
                        component_history.setdefault(name, []).append(value)
                # Update model parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()

            # Mean loss of each training step
            loss = torch.stack(loss_history).mean()

            # Plot
            if self.config.verbose:
                tqdm.write('Plotting...')

            self.writer.update_loss(loss, epoch_i, 'loss_epoch')
            epoch_components = {name: float(np.mean(values)) for name, values in component_history.items()}
            for name, value in epoch_components.items():
                self.writer.update_loss(value, epoch_i, name)
            if not os.path.exists(self.config.save_dir):
                os.makedirs(self.config.save_dir)
            self.train_losses.append(float(loss.detach().cpu().item()))
            if self.config.save_checkpoints:
                self.save_checkpoint(epoch_i)

            metrics = None
            if self.config.protocol == 'paper':
                self.evaluate(epoch_i)
            elif self.config.selection_metric == 'val_fscore':
                metrics = self.evaluate_metrics(epoch_i, self.val_loader,
                                                self.config.score_dir.joinpath('val'))
                self.val_metrics.append(metrics)
                self.writer.update_loss(metrics['fscore'], epoch_i, 'val_fscore')
                patience = self.config.early_stop_patience
                if patience > 0:
                    best_epoch = int(np.argmax([m['fscore'] for m in self.val_metrics]))
                    if epoch_i - best_epoch >= patience:
                        break

            log_parts = [f'Epoch {epoch_i + 1:03d}/{self.config.n_epochs:03d}',
                         f'loss={float(loss.detach().cpu().item()):.4f}']
            for name in ('loss_mse', 'loss_bce', 'loss_rank'):
                if name in epoch_components:
                    label = name.replace('loss_', '')
                    log_parts.append(f'{label}={epoch_components[name]:.4f}')
            if metrics is not None:
                log_parts.append(f'val_F1={metrics["fscore"]:.4f}')
                tau = metrics.get('kendall_tau')
                if tau is not None:
                    log_parts.append(f'val_Tau={tau:.4f}')
                rho = metrics.get('spearman_rho')
                if rho is not None:
                    log_parts.append(f'val_Rho={rho:.4f}')
                best_val = max(m['fscore'] for m in self.val_metrics)
                log_parts.append(f'best_val_F1={best_val:.4f}')
            print(' | '.join(log_parts), flush=True)

        if self.writer is not None:
            self.writer.close()
        if self.config.protocol == 'paper':
            return None
        selected_epoch = self.select_epoch()
        self.save_training_summary(selected_epoch)
        return selected_epoch

    def checkpoint_path(self, epoch_i):
        """Return the checkpoint path for an epoch."""
        return self.config.save_dir.joinpath(f'epoch-{epoch_i}.pt')

    def save_checkpoint(self, epoch_i):
        """Save model parameters for later clean-protocol model selection."""
        ckpt_path = self.checkpoint_path(epoch_i)
        torch.save(self.model.state_dict(), ckpt_path)
        ckpt_path.chmod(0o777)

    def load_checkpoint(self, epoch_i):
        """Load model parameters from a saved checkpoint."""
        ckpt_path = self.checkpoint_path(epoch_i)
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.config.device))

    def select_epoch(self):
        """Select the checkpoint epoch without looking at the test set."""
        if not self.val_metrics:
            raise ValueError('val_fscore selection requires validation metrics.')
        return int(np.argmax(np.asarray([m['fscore'] for m in self.val_metrics])))

    def save_training_summary(self, selected_epoch):
        """Persist clean/paper selection metadata for reproducibility."""
        summary = {
            'protocol': self.config.protocol,
            'supervision_setting': self.config.supervision_setting,
            'weak_label_mode': getattr(self.config, 'weak_label_mode', None),
            'weak_pos_ratio': getattr(self.config, 'weak_pos_ratio', None),
            'weak_neg_ratio': getattr(self.config, 'weak_neg_ratio', None),
            'weak_rank_margin': getattr(self.config, 'weak_rank_margin', None),
            'weak_rank_weight': getattr(self.config, 'weak_rank_weight', None),
            'selection_metric': self.config.selection_metric,
            'selected_epoch': int(selected_epoch),
            'selected_train_loss': self.train_losses[selected_epoch],
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'requested_batch_size': self.config.batch_size,
            'effective_batch_size': min(self.config.batch_size, len(self.train_loader)) if self.train_loader is not None else None,
        }
        if self.val_metrics:
            summary['selected_val_metrics'] = self.val_metrics[selected_epoch]
        if self.train_loader is not None:
            train_dataset = getattr(self.train_loader, 'dataset', self.train_loader)
            summary['train_keys'] = list(getattr(train_dataset, 'dataset_keys', []))
        if self.val_loader is not None:
            summary['val_keys'] = list(getattr(self.val_loader, 'dataset_keys', []))
        summary_path = self.config.save_dir.joinpath('selection.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        summary_path.chmod(0o777)

    def evaluate(self, epoch_i, save_weights=False, score_dir=None, loader=None):
        """Save frame importance scores in json format for val/test loaders."""
        eval_loader = loader if loader is not None else self.test_loader
        if eval_loader is None:
            raise ValueError('An evaluation loader is required.')
        self.model.eval()

        score_dir = Path(score_dir) if score_dir is not None else self.config.score_dir
        if not os.path.exists(score_dir):
            os.makedirs(score_dir)
        weights_save_path = score_dir.joinpath("weights.h5")
        out_scores_dict = {}
        for frame_features, video_name in tqdm(eval_loader, desc='Evaluate', ncols=80, leave=False):
            frame_features = frame_features.view(-1, self.config.input_size).to(self.config.device)

            with torch.no_grad():
                scores, attn_weights = self.model(frame_features)  # [1, seq_len]
                scores = scores.squeeze(0).cpu().numpy().tolist()
                attn_weights = attn_weights.cpu().numpy()
                out_scores_dict[video_name] = scores

            if save_weights:
                with h5py.File(weights_save_path, 'a') as weights:
                    weights.create_dataset(f"{video_name}/epoch_{epoch_i}", data=attn_weights)

        scores_save_path = score_dir.joinpath(f"{self.config.video_type}_{epoch_i}.json")
        with open(scores_save_path, 'w') as f:
            if self.config.verbose:
                tqdm.write(f'Saving score at {str(scores_save_path)}.')
            json.dump(out_scores_dict, f)
        scores_save_path.chmod(0o777)
        return scores_save_path

    def evaluate_metrics(self, epoch_i, loader, score_dir):
        """Evaluate a validation/test loader and return F1 for checkpoint selection."""
        scores_path = self.evaluate(epoch_i, score_dir=score_dir, loader=loader)
        metrics = self.compute_metrics(scores_path, include_rank=False)
        metrics['epoch'] = int(epoch_i)
        metrics_path = Path(score_dir).joinpath('metrics.jsonl')
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        return metrics

    def compute_metrics(self, scores_path, include_rank=True):
        """Compute official F1 and, for final evaluation, strict rank metrics."""
        eval_method = 'avg' if self.config.video_type.lower() == 'tvsum' else 'max'
        dataset_dir = 'SumMe' if self.config.video_type.lower() == 'summe' else 'TVSum'
        dataset_path = Path(self.config.dataset_root) / dataset_dir / (
            f'eccv16_dataset_{self.config.video_type.lower()}_google_pool5.h5'
        )
        with open(scores_path) as f:
            data = json.loads(f.read())
        keys = list(data.keys())
        all_scores = [np.asarray(data[video_name]) for video_name in keys]

        all_user_summary, all_shot_bound, all_nframes, all_positions = [], [], [], []
        all_f_scores = []
        with h5py.File(dataset_path, 'r') as hdf:
            for video_name, scores in zip(keys, all_scores):
                video_index = video_name[6:]
                user_summary = np.array(hdf.get('video_' + video_index + '/user_summary'))
                sb = np.array(hdf.get('video_' + video_index + '/change_points'))
                n_frames = np.array(hdf.get('video_' + video_index + '/n_frames'))
                positions = np.array(hdf.get('video_' + video_index + '/picks'))

                all_user_summary.append(user_summary)
                all_shot_bound.append(sb)
                all_nframes.append(n_frames)
                all_positions.append(positions)

        all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions)
        for video_index in range(len(all_summaries)):
            summary = all_summaries[video_index]
            user_summary = all_user_summary[video_index]
            all_f_scores.append(evaluate_summary(summary, user_summary, eval_method))

        metrics = {'fscore': float(np.mean(all_f_scores))}
        if include_rank:
            rank_metrics = compute_rank_metrics(
                scores_path,
                self.config.video_type,
                dataset_path=dataset_path,
                tvsum_anno_path=getattr(self.config, 'tvsum_anno_path', None),
            )
            metrics.update({
                'kendall_tau': rank_metrics['kendall_tau'],
                'spearman_rho': rank_metrics['spearman_rho'],
                'rank_metrics': rank_metrics,
            })
        else:
            metrics.update({
                'kendall_tau': None,
                'spearman_rho': None,
                'rank_metrics': {'computed': False, 'reason': 'validation checkpoint selection uses F-score only'},
            })
        return metrics


if __name__ == '__main__':
    pass
