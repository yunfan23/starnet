import argparse
import importlib
import os
import time
from pdb import set_trace
from shutil import copy2

import pdb
import torch
import torch.distributed
import yaml
from tensorboardX import SummaryWriter
from torch.backends import cudnn

from trainers.utils.utils import ForkedPdb


def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='Point Cloud Generation Experiment')
    parser.add_argument('config', type=str, help='The configuration file.')

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
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all '
                             'available GPUs.')

    # Resume:
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--pretrained', default=None, type=str,
                        help="Pretrained checkpoint")

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

    #  Create log_name
    cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    # Currently save dir and log_dir are the same
    config.log_name = "logs/%s_%s" % (cfg_file_name, run_time)
    config.save_dir = "logs/%s_%s" % (cfg_file_name, run_time)
    config.log_dir = "logs/%s_%s" % (cfg_file_name, run_time)
    os.makedirs(config.log_dir+'/config')
    os.makedirs(config.log_dir+'/checkpoints')
    os.makedirs(config.log_dir+'/image')
    os.makedirs(config.log_dir+'/val')
    os.makedirs(config.log_dir+'/models')
    copy2(args.config, config.log_dir+'/config')
    # model_gen = config.models.gen.type
    # model_gen = model_gen.replace(".", "/") + ".py"
    # copy2(model_gen, config.log_dir+'/models')
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

    start_epoch = 0

    if args.resume:
        if args.pretrained is not None:
            start_epoch = trainer.resume(args.pretrained, strict=True)
            val_info = trainer.validate(test_loader, idx=-1)
        else:
            start_epoch = trainer.resume(cfg.resume.dir)

    # If test run, go through the validation loop first
    if args.test_run:
        trainer.save(epoch=-1, step=-1)
        val_info = trainer.validate(test_loader, epoch=-1)
        trainer.log_val(val_info, writer=writer, epoch=-1)

    # main training loop
    print("Start epoch: %d End epoch: %d" % (start_epoch, cfg.trainer.epochs))
    step = 0
    logs_info = {}
    for epoch in range(start_epoch, cfg.trainer.epochs):
        visualize = epoch % int(cfg.viz.viz_freq) == 0
        # pdb.set_trace()
        start_time = time.time()
        logs_info = trainer.update(train_loader)
        duration = time.time() - start_time

        if epoch % int(cfg.viz.log_freq) == 0 and int(cfg.viz.log_freq) > 0:
            trainer.log_train(logs_info, test_loader, writer=writer, epoch=epoch, visualize=visualize)

        if epoch % int(cfg.viz.save_freq) == 0 and int(cfg.viz.save_freq) > 0:
            trainer.save(epoch=epoch, step=step)

        # if epoch % int(cfg.viz.val_freq) == 0:
        if epoch % int(cfg.viz.val_freq) == 0 and int(cfg.viz.val_freq) > 0:
            val_info = trainer.validate(test_loader, epoch)
            trainer.log_val(val_info, writer=writer, epoch=epoch)

        # pdb.set_trace()
        trainer.epoch_end(epoch, writer=writer)
        if epoch % int(cfg.viz.log_freq) == 0 and int(cfg.viz.log_freq) > 0:
            if 'loss' in logs_info.keys():
                loss = logs_info['loss']
                print(f"[INFO] Epoch {epoch:-4} Time {duration:.1f}s: Loss {loss:.4f}")
            else:
                print(f"[INFO] Epoch {epoch:-4} Time {duration:.1f}s")
    writer.close()


if __name__ == '__main__':
    # command line args
    args, cfg = get_args()

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfg)

    main_worker(cfg, args)
