# flake8: noqa
import argparse
import os
import os.path as osp
import torch
import yaml

import ASR21cm.archs
import ASR21cm.data
import ASR21cm.losses
import ASR21cm.models
from basicsr.train import train_pipeline


def _pre_create_exp_dirs(root_path):
    """Ensure the experiment directory exists before BasicSR's train_pipeline
    runs. BasicSR skips mkdir when resume_state is set, which causes a crash
    when the experiment name differs from the resumed checkpoint.

    Also symlinks model checkpoints from the source experiment into the new
    experiment's models dir, because check_resume unconditionally rewrites
    pretrain_network_* to point at {current_experiment}/models/net_*_{iter}.pth.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str)
    args, _ = parser.parse_known_args()
    if args.opt is None:
        return
    with open(args.opt, 'r') as f:
        opt = yaml.safe_load(f)

    exp_root = osp.join(root_path, 'experiments', opt['name'])
    for subdir in ('', 'models', 'training_states', 'visualization'):
        os.makedirs(osp.join(exp_root, subdir), exist_ok=True)

    # If resuming from a different experiment, symlink its model checkpoints
    # into the new experiment's models dir so check_resume can find them.
    resume_state_path = opt.get('path', {}).get('resume_state')
    if resume_state_path and osp.exists(resume_state_path):
        src_models = osp.dirname(osp.dirname(resume_state_path))  # .../ExperimentName
        src_models = osp.join(src_models, 'models')
        dst_models = osp.join(exp_root, 'models')
        if osp.isdir(src_models) and osp.realpath(src_models) != osp.realpath(dst_models):
            for fname in os.listdir(src_models):
                src = osp.join(src_models, fname)
                dst = osp.join(dst_models, fname)
                if not osp.exists(dst):
                    os.symlink(src, dst)


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    _pre_create_exp_dirs(root_path)
    train_pipeline(root_path)
