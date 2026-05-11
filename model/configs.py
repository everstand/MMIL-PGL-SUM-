# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import torch
import pprint

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / 'data' / 'datasets'
DEFAULT_SPLITS_ROOT = DEFAULT_DATASET_ROOT / 'splits'
DEFAULT_EXP_ROOT = REPO_ROOT / 'experiments' / 'results' / 'exp1'
DEFAULT_WEAK_LABELS_ROOT = REPO_ROOT / 'data' / 'weak_labels'


def str2bool(v):
    """ Transcode string to boolean.

    :param str v: String to be transcoded.
    :return: The boolean transcoding of the string.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        self.log_dir, self.score_dir, self.save_dir = None, None, None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.exp_root = Path(self.exp_root)
        self.dataset_root = Path(self.dataset_root)
        self.splits_root = Path(self.splits_root)
        self.tvsum_anno_path = Path(self.tvsum_anno_path)
        self.weak_labels_root = Path(self.weak_labels_root)

        self.protocol = self.protocol.lower()
        if self.protocol not in ('paper', 'clean'):
            raise ValueError("protocol must be either 'paper' or 'clean'.")
        if self.selection_metric != 'val_fscore':
            raise ValueError("selection_metric must be val_fscore.")
        if self.protocol == 'clean' and not self.save_checkpoints:
            raise ValueError("clean protocol requires --save_checkpoints true.")
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError("val_ratio must be in (0, 1).")
        if self.early_stop_patience < 0:
            raise ValueError("early_stop_patience must be >= 0.")
        self.supervision_setting = self.supervision_setting.lower()
        if self.supervision_setting not in ('supervised', 'weak'):
            raise ValueError("supervision_setting must be either 'supervised' or 'weak'.")
        self.weak_labels_path = None
        self.weak_target_norm = self.weak_target_norm.lower()
        if self.weak_target_norm not in ('none', 'per_video_minmax'):
            raise ValueError("weak_target_norm must be one of: none, per_video_minmax.")
        self.reg_loss = self.reg_loss.lower()
        if self.reg_loss not in ('mse', 'huber'):
            raise ValueError("reg_loss must be one of: mse, huber.")
        if self.lambda_step_reg < 0.0:
            raise ValueError("lambda_step_reg must be >= 0.")
        if self.lambda_shot_aux < 0.0:
            raise ValueError("lambda_shot_aux must be >= 0.")
        self.shot_pool = self.shot_pool.lower()
        if self.shot_pool not in ('mean',):
            raise ValueError("shot_pool must be one of: mean.")
        if self.lambda_rankcorr < 0.0:
            raise ValueError("lambda_rankcorr must be >= 0.")
        if self.rankcorr_tau <= 0.0:
            raise ValueError("rankcorr_tau must be > 0.")
        if not 0.0 < self.weak_pos_ratio < 1.0:
            raise ValueError("weak_pos_ratio must be in (0, 1).")
        if not 0.0 < self.weak_neg_ratio < 1.0:
            raise ValueError("weak_neg_ratio must be in (0, 1).")
        if self.weak_pos_ratio + self.weak_neg_ratio >= 1.0:
            raise ValueError("weak_pos_ratio + weak_neg_ratio must be < 1.0.")
        if self.weak_rank_margin < 0.0:
            raise ValueError("weak_rank_margin must be >= 0.")
        if self.weak_rank_weight < 0.0:
            raise ValueError("weak_rank_weight must be >= 0.")
        if self.supervision_setting == 'weak':
            self.weak_labels_path = self.weak_labels_root / f'{self.video_type.lower()}_shot_utility.npy'
            if not self.weak_labels_path.exists():
                raise FileNotFoundError(f'Weak label file not found: {self.weak_labels_path}')

        self.set_dataset_dir(self.video_type)

    def set_dataset_dir(self, video_type='TVSum'):
        """ Function that sets as class attributes the necessary directories for logging important training information.

        :param str video_type: The Dataset being used, SumMe or TVSum.
        """
        self.log_dir = self.exp_root.joinpath(video_type, 'logs/split' + str(self.split_index))
        self.score_dir = self.exp_root.joinpath(video_type, 'results/split' + str(self.split_index))
        self.save_dir = self.exp_root.joinpath(video_type, 'models/split' + str(self.split_index))

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """ Get configurations as attributes of class
        1. Parse configurations with argparse.
        2. Create Config class initialized with parsed kwargs.
        3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train', help='Mode for the configuration [train | val | test]')
    parser.add_argument('--verbose', type=str2bool, default='false', help='Print or not training messages')
    parser.add_argument('--video_type', type=str, default='SumMe', help='Dataset to be used')
    parser.add_argument('--protocol', type=str, default='clean', choices=['paper', 'clean'],
                        help='Default mainline protocol is clean; use paper only when explicitly reproducing legacy test-in-selection behavior.')
    parser.add_argument('--exp_root', type=str, default=str(DEFAULT_EXP_ROOT),
                        help='Root directory where experiment logs, scores and checkpoints are stored')
    parser.add_argument('--dataset_root', type=str, default=str(DEFAULT_DATASET_ROOT),
                        help='Root directory containing dataset feature files and annotations')
    parser.add_argument('--splits_root', type=str, default=str(DEFAULT_SPLITS_ROOT),
                        help='Root directory containing split JSON files')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation ratio carved from outer-fold train_keys for clean protocol')
    parser.add_argument('--val_seed', type=int, default=0,
                        help='Seed for deterministic inner validation split')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='Stop after this many epochs without val F-score improvement; 0 disables early stopping')
    parser.add_argument(
        '--tvsum_anno_path',
        type=str,
        default=str(DEFAULT_DATASET_ROOT / 'TVSum' / 'ydata-anno.tsv'),
        help='Path to official TVSum ydata annotation TSV for rank metrics'
    )
    parser.add_argument('--supervision_setting', type=str, default='weak',
                        choices=['supervised', 'weak'],
                        help='Training supervision paradigm; weak requires external weak labels and never uses gtscore during training')
    parser.add_argument('--weak_labels_root', type=str, default=str(DEFAULT_WEAK_LABELS_ROOT),
                        help='Directory containing external weak label files such as summe_shot_utility.npy and tvsum_shot_utility.npy')
    parser.add_argument('--weak_target_norm', type=str, default='per_video_minmax',
                        choices=['none', 'per_video_minmax'])
    parser.add_argument('--reg_loss', type=str, default='huber',
                        choices=['mse', 'huber'])
    parser.add_argument('--lambda_step_reg', type=float, default=1.0)
    parser.add_argument('--use_shot_balance', type=str2bool, default='true')
    parser.add_argument('--lambda_shot_aux', type=float, default=0.5)
    parser.add_argument('--shot_pool', type=str, default='mean', choices=['mean'])
    parser.add_argument('--use_rankcorr', type=str2bool, default='true')
    parser.add_argument('--lambda_rankcorr', type=float, default=0.2)
    parser.add_argument('--rankcorr_tau', type=float, default=0.1)
    parser.add_argument('--weak_pos_ratio', type=float, default=0.15,
                        help='Top ratio of externally weak-labeled positions treated as positive for pairwise ranking')
    parser.add_argument('--weak_neg_ratio', type=float, default=0.15,
                        help='Bottom ratio of externally weak-labeled positions treated as negative for pairwise ranking')
    parser.add_argument('--weak_rank_margin', type=float, default=0.1,
                        help='Margin used by the weak pairwise ranking loss')
    parser.add_argument('--weak_rank_weight', type=float, default=1.0,
                        help='Weight applied to the weak pairwise ranking loss term')

    # Model
    parser.add_argument('--input_size', type=int, default=1024, help='Feature size expected in the input')
    parser.add_argument('--seed', type=int, default=12345, help='Chosen seed for generating random numbers')
    parser.add_argument('--fusion', type=str, default="add", help="Type of feature fusion")
    parser.add_argument('--n_segments', type=int, default=4, help='Number of segments to split the video')
    parser.add_argument('--pos_enc', type=str, default="absolute", help="Type of pos encoding [absolute|relative|None]")
    parser.add_argument('--heads', type=int, default=8, help="Number of global heads for the attention module")

    # Train
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=20, help='Size of each batch in training')
    parser.add_argument('--clip', type=float, default=5.0, help='Max norm of the gradients')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate used for the modules')
    parser.add_argument('--l2_req', type=float, default=1e-5, help='Regularization factor')
    parser.add_argument('--split_index', type=int, default=0, help='Data split to be used [0-4]')
    parser.add_argument('--init_type', type=str, default="xavier", help='Weight initialization method')
    parser.add_argument('--init_gain', type=float, default=None, help='Scaling factor for the initialization methods')
    parser.add_argument('--save_checkpoints', type=str2bool, default='true',
                        help='Save model parameters after each training epoch')
    parser.add_argument('--selection_metric', type=str, default='val_fscore',
                        choices=['val_fscore'],
                        help='Checkpoint selection metric; clean protocol uses validation F-score')

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()
    import ipdb
    ipdb.set_trace()
