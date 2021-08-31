# Copyright (c) Yunfan Zhang All Rights Reserved.
# File Name: ae_trainer.py
# Author: Yunfan Zhang
# Mail: yunfan.zhang23@gmail.com
# Github: https://github.com/yunfan23
# Blog: http://www.zhangyunfan.tech/
# Created Time: 2021-08-02

import os
import tqdm
import torch
import importlib
import numpy as np
from trainers.base_trainer import BaseTrainer
from trainers.utils.vis_utils import visualize_point_clouds_3d, \
    visualize_procedure, visualize_point_clouds_img
from trainers.utils.utils import get_opt, get_prior, \
    ground_truth_reconstruct_multi, set_random_seed, visualize_point_clouds_3d

import evaluation.emd.emd_module as emd


try:
    from evaluation.evaluation_metrics import EMD_CD
    eval_reconstruciton = True
except:  # noqa
    # Skip evaluation
    eval_reconstruciton = False


class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        # The networks
        decoder_lib = importlib.import_module(cfg.models.decoder.type)
        self.decoder = decoder_lib.Generator(cfg.models.decoder)
        self.decoder.cuda()
        print("Decoder:")
        print(self.decoder)

        self.loss = emd.emdModule().cuda()

        encoder_lib = importlib.import_module(cfg.models.encoder.type)
        self.encoder = encoder_lib.Encoder(cfg.models.encoder)
        self.encoder.cuda()
        print("Encoder:")
        print(self.encoder)

        # The optimizer
        if not (hasattr(self.cfg.trainer, "opt_enc") and
                hasattr(self.cfg.trainer, "opt_dec")):
            self.cfg.trainer.opt_enc = self.cfg.trainer.opt
            self.cfg.trainer.opt_dec = self.cfg.trainer.opt

        self.opt_enc, self.scheduler_enc = get_opt(
            self.encoder.parameters(), self.cfg.trainer.opt_enc)
        self.opt_dec, self.scheduler_dec = get_opt(
            self.decoder.parameters(), self.cfg.trainer.opt_dec)

        # Sigmas
        if hasattr(cfg.trainer, "sigmas"):
            self.sigmas = cfg.trainer.sigmas
        else:
            self.sigma_begin = float(cfg.trainer.sigma_begin)
            self.sigma_end = float(cfg.trainer.sigma_end)
            self.num_classes = int(cfg.trainer.sigma_num)
            self.sigmas = np.exp(
                np.linspace(np.log(self.sigma_begin),
                            np.log(self.sigma_end),
                            self.num_classes))
        print("Sigma:, ", self.sigmas)

        # Prepare save directory
        os.makedirs(os.path.join(cfg.save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "val"), exist_ok=True)

        # Prepare variable for summy
        self.oracle_res = None

    def multi_gpu_wrapper(self, wrapper):
        self.encoder = wrapper(self.encoder)
        self.deocder = wrapper(self.deocder)

    def epoch_end(self, epoch, writer=None, **kwargs):
        if self.scheduler_dec is not None:
            self.scheduler_dec.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_dec_lr', self.scheduler_dec.get_lr()[0], epoch)
        if self.scheduler_enc is not None:
            self.scheduler_enc.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_enc_lr', self.scheduler_enc.get_lr()[0], epoch)

    def update(self, data, *args, **kwargs):
        if 'no_update' in kwargs:
            no_update = kwargs['no_update']
        else:
            no_update = False
        if not no_update:
            self.encoder.train()
            self.decoder.train()
            self.opt_enc.zero_grad()
            self.opt_dec.zero_grad()

        tr_pts = data['tr_points'].cuda()  # (B, #points, 3)smn_ae_trainer.py
        batch_size = tr_pts.size(0)
        z_mu, z_sigma = self.encoder(tr_pts)
        z = z_mu + 0 * z_sigma
        reconv_pts = self.decoder(z)
        reconv_pts = reconv_pts.transpose(1, 2)
        # import pdb; pdb.set_trace()
        loss, _ = self.loss(reconv_pts, tr_pts, eps=0.005, iters=50)
        loss = torch.sqrt(loss).mean(1).mean()
        loss.backward()
        self.opt_enc.step()
        self.opt_dec.step()

        return {
            'loss': loss.detach().cpu().item()
        }

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        if writer is None:
            return

        # Log training information to tensorboard
        train_info = {k: (v.cpu() if not isinstance(v, float) else v)
                      for k, v in train_info.items()}
        for k, v in train_info.items():
            if not ('loss' in k):
                continue
            if step is not None:
                writer.add_scalar('train/' + k, v, step)
            else:
                assert epoch is not None
                writer.add_scalar('train/' + k, v, epoch)

        if visualize:
            with torch.no_grad():
                print("Visualize: %s" % step)
                gtr = train_data['te_points']  # ground truth point cloud
                inp = train_data['tr_points']  # input for encoder
                num_vis = min(
                    getattr(self.cfg.viz, "num_vis_samples", 5),
                    gtr.size(0))

                print("Recon:")
                rec = self.reconstruct(inp[:num_vis].cuda(), num_points=inp.size(1))
                rec = rec.transpose(1, 2)
                all_imgs = []
                # import pdb; pdb.set_trace()
                for idx in range(num_vis):
                    img = visualize_point_clouds_3d([rec[idx], gtr[idx]], ["recon", "shape"])
                    all_imgs.append(img)
                img = np.concatenate(all_imgs, axis=1)
                writer.add_image(
                    'tr_vis/overview', torch.as_tensor(img), step)


    def validate(self, test_loader, epoch, *args, **kwargs):
        if not eval_reconstruciton:
            return {}

        print("Validation (reconstruction):")
        all_ref, all_rec, all_smp, all_ref_denorm = [], [], [], []
        all_rec_gt, all_inp_denorm, all_inp = [], [], []
        for data in tqdm.tqdm(test_loader):
            ref_pts = data['te_points'].cuda()
            inp_pts = data['tr_points'].cuda()
            m = data['mean'].cuda()
            std = data['std'].cuda()
            rec_pts, _ = self.reconstruct(inp_pts, num_points=inp_pts.size(1))

            # denormalize
            inp_pts_denorm = inp_pts.clone() * std + m
            ref_pts_denorm = ref_pts.clone() * std + m
            rec_pts = rec_pts * std + m

            all_inp.append(inp_pts)
            all_inp_denorm.append(inp_pts_denorm.view(*inp_pts.size()))
            all_ref_denorm.append(ref_pts_denorm.view(*ref_pts.size()))
            all_rec.append(rec_pts.view(*ref_pts.size()))
            all_ref.append(ref_pts)

        inp = torch.cat(all_inp, dim=0)
        rec = torch.cat(all_rec, dim=0)
        ref = torch.cat(all_ref, dim=0)
        ref_denorm = torch.cat(all_ref_denorm, dim=0)
        inp_denorm = torch.cat(all_inp_denorm, dim=0)
        for name, arr in [
            ('inp', inp), ('rec', rec), ('ref', ref),
            ('ref_denorm', ref_denorm), ('inp_denorm', inp_denorm)]:
            np.save(
                os.path.join(
                    self.cfg.save_dir, 'val', '%s_ep%d.npy' % (name, epoch)),
                arr.detach().cpu().numpy()
            )
        all_res = {}

        # Oracle CD/EMD, will compute only once
        if self.oracle_res is None:
            rec_res = EMD_CD(inp_denorm, ref_denorm, 1)
            rec_res = {
                ("val/rec/%s" % k): (v if isinstance(v, float) else v.item())
                for k, v in rec_res.items()}
            all_res.update(rec_res)
            print("Validation oracle (denormalize) Epoch:%d " % epoch, rec_res)
            self.oracle_res = rec_res
        else:
            all_res.update(self.oracle_res)

        # Reconstruction CD/EMD
        all_res = {}
        rec_res = EMD_CD(rec, ref_denorm, 1)
        rec_res = {
            ("val/rec/%s" % k): (v if isinstance(v, float) else v.item())
            for k, v in rec_res.items()}
        all_res.update(rec_res)
        print("Validation Recon (denormalize) Epoch:%d " % epoch, rec_res)

        return all_res

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'opt_enc': self.opt_enc.state_dict(),
            'opt_dec': self.opt_dec.state_dict(),
            'dec': self.decoder.state_dict(),
            'enc': self.encoder.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)

    def resume(self, path, strict=True, **kwargs):
        ckpt = torch.load(path)
        self.encoder.load_state_dict(ckpt['enc'], strict=strict)
        self.decoder.load_state_dict(ckpt['dec'], strict=strict)
        self.opt_enc.load_state_dict(ckpt['opt_enc'])
        self.opt_dec.load_state_dict(ckpt['opt_dec'])
        start_epoch = ckpt['epoch']
        return start_epoch

    def sample(self, num_shapes=1, num_points=2048):
        with torch.no_grad():
            z = torch.randn(num_shapes, self.cfg.models.encoder.zdim).cuda()
            samples = self.decoder(z)
            visualize_point_clouds_3d(samples)
            return samples

    def reconstruct(self, data, num_points=2048):
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            inps = data['tr_points'].cuda()
            z, _ = self.encoder(inps)
            samples = self.decoder(z)
            samples = samples.transpose(1, 2)
            visualize_point_clouds_img(inps.cpu().numpy(), samples.cpu().numpy())

