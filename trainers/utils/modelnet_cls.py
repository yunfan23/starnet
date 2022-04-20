import argparse
import importlib
import tqdm
import os
import torch
import time
import numpy as np
from pprint import pprint
from shutil import copy2
from tensorboardX import SummaryWriter
import torch.nn as nn
import yaml
from torch.backends import cudnn
import pdb
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from trainers.utils import provider


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
    # writer = SummaryWriter(logdir=cfg.log_name)
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args)
    trainer.resume(args.pretrained)
    data_lib = importlib.import_module(cfg.data.type)
    dataset = data_lib.get_data_loaders(cfg)
    training = dataset['train_loader']
    test = dataset['test_loader']
    
    label_training = []
    feat_training = []
    for raw in training:
        label_training.append(raw['cate_idx'])
        data_training = raw['train_points']
        feat_training.append(trainer.get_feat(data_training.cuda()))
    for raw in training:
        label_training.append(raw['cate_idx'])
        data_training = raw['train_points']
        feat_training.append(trainer.get_feat(data_training.cuda()))
    for raw in training:
        label_training.append(raw['cate_idx'])
        data_training = raw['train_points']
        feat_training.append(trainer.get_feat(data_training.cuda()))

    label_training = torch.cat(label_training, dim=0).numpy()
    feat_training = torch.cat(feat_training, dim=0).cpu().numpy()
    # print(len(label_training))
    # print(len(feat_training))
    # standardScaler = StandardScaler()
    # standardScaler.fit(feat_training)
    # feat_training_std = standardScaler.transform(feat_training)
    # svc = LinearSVC(random_state=0, C=1e-5, max_iter=1000, fit_intercept=True, intercept_scaling=2,penalty='l2', loss='hinge', dual=True)
    # svc = LinearSVC(random_state=0, C=0.1, max_iter=1000, fit_intercept=True, intercept_scaling=2,penalty='l2', loss='hinge', dual=True)
    # svc = LinearSVC(C=0.2, max_iter=100000)
    # svc = LinearSVC(C=0.001, loss='hinge', max_iter=100000)
    svc = LinearSVC(C=0.001, max_iter=100000)
    # svc = LinearSVC(C=0.001, max_iter=100000)
    # svc.fit(feat_training_std, label_training)
    svc.fit(feat_training, label_training)
    # pdb.set_trace()
    
    label_test = []
    feat_test = []
    for raw in test:
        label_test.append(raw['cate_idx'])
        data_test = raw['train_points']
        feat_test.append(trainer.get_feat(data_test.cuda()))

    label_test = torch.cat(label_test, dim=0).numpy()
    feat_test = torch.cat(feat_test, dim=0).cpu().numpy()
    # print(len(label_test))
    # print(len(feat_test))
    # standardScaler.fit(feat_test)
    # feat_test_std = standardScaler.transform(feat_test)
    predict = svc.predict(feat_test)
    ac_score = metrics.accuracy_score(label_test, predict)
    cls_report = metrics.classification_report(label_test, predict)
    print(ac_score)
    # print(cls_report)


if __name__ == '__main__':
    # command line args
    args, cfg = get_args()

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfg)

    main_worker(cfg, args)
