# -*- coding: utf-8 -*-
"""Run model/main.py with faster TensorBoard scalar flushing.

This keeps core training files unchanged while allowing external monitors to
read per-epoch loss promptly from TensorBoard event files.
"""
import inspect
import runpy
import sys
from pathlib import Path

import tensorboardX

_original_init = tensorboardX.SummaryWriter.__init__


def _fast_flush_init(self, *args, **kwargs):
    params = inspect.signature(_original_init).parameters
    if 'flush_secs' in params and 'flush_secs' not in kwargs:
        kwargs['flush_secs'] = 1
    if 'max_queue' in params and 'max_queue' not in kwargs:
        kwargs['max_queue'] = 1
    return _original_init(self, *args, **kwargs)


tensorboardX.SummaryWriter.__init__ = _fast_flush_init

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / 'model'))
sys.argv = [str(repo_root / 'model' / 'main.py')] + sys.argv[1:]
runpy.run_path(str(repo_root / 'model' / 'main.py'), run_name='__main__')
