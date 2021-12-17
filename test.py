import argparse
import importlib
import tqdm
import os
import time
import numpy as np
from pprint import pprint
from shutil import copy2
from tensorboardX import SummaryWriter
import torch.nn as nn
import yaml
from torch.backends import cudnn
import pdb


def get_args():
    # command line args
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('config', type=str,
                        help='The configuration file.')

    # distributed training
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to '
                             'launch N processes per node, which has N GPUs. '
                             'This is the fastest way to use PyTorch for '
                             'either single node or multi node data parallel '
                             'training')

    # Resume:
    parser.add_argument('--pretrained', default=None, type=str,
                        help="Pretrained cehckpoint")

    # Evaluation split
    parser.add_argument('--eval_split', default='val', type=str,
                        help="The split to be evaluated.")
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

    # #  Create log_name
    cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    # Currently save dir and log_dir are the same
    config.log_name = "logs/%s_val_%s" % (cfg_file_name, run_time)
    config.save_dir = "logs/%s_val_%s" % (cfg_file_name, run_time)
    config.log_dir = "logs/%s_val_%s" % (cfg_file_name, run_time)
    os.makedirs(config.log_dir + '/config')
    os.makedirs(config.log_dir + '/val')
    copy2(args.config, config.log_dir + '/config')
    return args, config


def main_worker(cfg, args):
    # basic setup
    cudnn.benchmark = True

    writer = SummaryWriter(logdir=cfg.log_name)
    data_lib = importlib.import_module(cfg.data.type)
    loaders = data_lib.get_data_loaders(cfg.data, args)
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args)

    # if args.distributed:  # Multiple processes, single GPU per process
    #     def wrapper(m):
    #         return nn.DataParallel(m)
    #     trainer.multi_gpu_wrapper(wrapper)
    trainer.resume(args.pretrained)
    idx = 0
    print(cfg.save_dir)
    # smp = "/home/yunfan/workarea/style-aware_pcgen.v1/logs/airplane_gen_v2_val_2021-Nov-09-12-43-44/val/smp_84.npy"
    # ref = "/home/yunfan/workarea/style-aware_pcgen.v1/logs/airplane_gen_v2_val_2021-Nov-09-12-43-44/val/ref_84.npy"
    # smp = "/home/yunfan/workarea/style-aware_pcgen.v1/logs/chair_gen_v2_val_2021-Nov-09-20-53-34/val/smp_0.npy"
    # ref = "/home/yunfan/workarea/style-aware_pcgen.v1/logs/chair_gen_v2_val_2021-Nov-09-20-53-34/val/ref_0.npy"
    # val_info = trainer.validate(test_loader, idx, evaluation=True, smp=smp, ref=ref)
    # pprint(val_info)
    # for idx in tqdm.tqdm(range(100), desc="idx"):
    # smp = '/home/yunfan/workarea/style-aware_pcgen.v1/logs/airplane_gen_v2_val_2021-Nov-12-16-53-54/val/smp_0.npy'
    # smp = './scratch/smp.npy'
    # ref = '/home/yunfan/workarea/style-aware_pcgen.v1/logs/airplane_gen_v2_val_2021-Nov-12-16-53-54/val/ref_0.npy'
    for idx in tqdm.tqdm(range(1), desc="idx"):
        val_info = trainer.validate(test_loader, idx, evaluation=True)
        # val_info = trainer.validate(test_loader, epoch=-1, evaluation=True)
        # val_info = trainer.validate(test_loader, idx, evaluation=False, smp=smp, ref=ref)
        # val_info = trainer.validate_fpd(test_loader, epoch=-1)
        # val_info = trainer.validate_fpd(train_loader, epoch=-1)
        # pprint(idx)
        # pprint(val_info)
    writer.close()


if __name__ == '__main__':
    # command line args
    args, cfg = get_args()

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfg)

    main_worker(cfg, args)
