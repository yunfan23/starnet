import argparse
import importlib
import os
import time
from pdb import set_trace
from shutil import copy2

import torch
import torch.distributed
import yaml
from tensorboardX import SummaryWriter
from torch.backends import cudnn


def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='Flow-based Point Cloud Generation Experiment')
    parser.add_argument('config', type=str,
                        help='The configuration file.')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to '
                             'launch N processes per node, which has N GPUs. '
                             'This is the fastest way to use PyTorch for '
                             'either single node or multi node data parallel '
                             'training')
    parser.add_argument('--num_obj', default=5, type=int,
                        help='number of generated object')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all '
                             'available GPUs.')

    # Resume:
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--pretrained', default=None, type=str,
                        help="Pretrained cehckpoint")

    # Test run:
    parser.add_argument('--test_run', default=False, action='store_true')
    args = parser.parse_args()

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    # parse config file

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    return args, config


def main_worker(cfg, args):
    # basic setup
    cudnn.benchmark = True
    data_lib = importlib.import_module(cfg.data.type)
    loaders = data_lib.get_data_loaders(cfg.data, args)
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args)

    start_epoch = trainer.resume(args.pretrained)

    # If test run, go through the validation loop first
    if args.test_run:
        trainer.trial(test_loader, writer, epoch=-1)
        return

    # main training loop
    print("start gen ...")
    # trainer.generate_v2(train_loader, vis=True)
    trainer.generate_v3(cfg.data.cates[0], vis=True)
    print("finish gen ...")


if __name__ == '__main__':
    # command line args
    args, cfg = get_args()
    main_worker(cfg, args)
