import importlib
import math
import os
import pdb
import random

import evaluation.emd.emd_module as emd
import numpy as np
import torch
import torch.nn as nn
import tqdm
from evaluation.chamfer_distance import ChamferDistanceMean
from evaluation.evaluation_metrics import (EMD_CD, compute_all_metrics,
                                           jsd_between_point_cloud_sets)
from Frechet.FPD import calculate_fpd

from trainers.ae_sparenet_trainer_3D import Trainer as BaseTrainer
from trainers.base_trainer import BaseTrainer
from trainers.utils.gan_losses import dis_loss, gen_loss, gradient_penalty
from trainers.utils.utils import (get_opt, get_prior,
                                  ground_truth_reconstruct_multi,
                                  normalize_point_clouds, set_random_seed,
                                  visualize_point_clouds_3d)
from trainers.utils.vis_utils import (visualize_point_clouds_3d,
                                      visualize_point_clouds_img,
                                      visualize_procedure)


class Trainer(BaseTrainer):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        # Now initialize the VAE part
        lenc_lib = importlib.import_module(cfg.models.lenc.type)
        self.lenc = lenc_lib.Encoder(cfg.models.lenc)
        self.lenc.cuda()
        print("Latent Encoder:")
        print(self.lenc)

        ldec_lib = importlib.import_module(cfg.models.ldec.type)
        self.ldec = ldec_lib.Decoder(cfg.models.ldec)
        self.ldec.cuda()
        print("Latent Decoder:")
        print(self.ldec)

        # Optimizers
        self.opt_lenc, self.scheduler_lenc = get_opt(self.lenc.parameters(), self.cfg.trainer.opt_lenc)
        self.opt_ldec, self.scheduler_ldec = get_opt(self.ldec.parameters(), self.cfg.trainer.opt_ldec)

        # load point cloud encoder
        enc_lib = importlib.import_module(cfg.models.encoder.type)
        self.enc = enc_lib.Encoder(cfg.models.encoder)
        self.enc.cuda()
        print("Point Cloud Encoder:")
        print(self.enc)
        print(self.cfg.trainer.ae_pretrained)
        ckpt = torch.load(self.cfg.trainer.ae_pretrained)
        self.enc.load_state_dict(ckpt['enc'], strict=True)

        dec_lib = importlib.import_module(cfg.models.decoder.type)
        self.dec = dec_lib.Decoder(cfg.models.decoder)
        self.dec.cuda()
        print("Point Cloud Decoder:")
        print(self.dec)
        self.dec.load_state_dict(ckpt['dec'], strict=True)

    def epoch_end(self, epoch, writer=None, **kwargs):
        super().epoch_end(epoch, writer=writer)

        if self.scheduler_ldec is not None:
            self.scheduler_ldec.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_ldec_lr', self.scheduler_ldec.get_lr()[0], epoch)

        if self.scheduler_lenc is not None:
            self.scheduler_lenc.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_lenc_lr', self.scheduler_lenc.get_lr()[0], epoch)

    def loss_fn(self, z_hat, z, mu, logvar, kld_weight=1.):
        import torch.nn.functional as F

        # recon_loss = -nn.functional.kl_div(z_hat, z)
        recon_loss = F.l1_loss(z_hat, z)
        # BCE = F.binary_cross_entropy(F.sigmoid(z_hat), F.sigmoid(z))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + KLD
        return loss

    def update(self, dataset, *args, **kwargs):
        total_loss = 0.
        avg_loss = 0.
        num_items = 0.
        for _, data in enumerate(dataset):
            self.lenc.train()
            self.ldec.train()
            self.opt_lenc.zero_grad()
            self.opt_ldec.zero_grad()
            self.enc.eval()
            
            tr_pts = data['tr_points'].cuda()
            bs = tr_pts.size(0)
            with torch.no_grad():
                z, _ = self.enc(tr_pts)
            mu, log_var = self.lenc(z)
            z_hat = self.ldec(self.reparameterize(mu, log_var))
            # import pdb; pdb.set_trace()
            loss = self.loss_fn(z_hat, z, mu, log_var)
            loss.backward()
            self.opt_lenc.step()
            self.opt_ldec.step()

            total_loss += loss.detach().cpu().item() * bs
            num_items += bs
            avg_loss = total_loss / num_items

        return {'loss': avg_loss}
        

    def log_train(self, train_info, train_data, writer=None, step=None, epoch=None, visualize=False, **kwargs):
        if writer is None:
            return
        # Log training information to tensorboard
        train_info = {k: (v.cpu() if not isinstance(v, float) else v)
                      for k, v in train_info.items()}
        for k, v in train_info.items():
            if not ('loss' in k):
                continue
            writer.add_scalar('train/' + k, v, epoch)

        if visualize:
            with torch.no_grad():
                print("Visualize generation results: %s" % epoch)
                data = next(iter(train_data))
                gtr = data['te_points']
                num_vis = min(getattr(self.cfg.viz, "num_vis_samples", 5), gtr.size(0))
                smp = self.generate(num_shapes=num_vis * 4)

                all_imgs = []
                for idx in range(num_vis):
                    img = visualize_point_clouds_3d([smp[idx], smp[idx + num_vis], smp[idx + num_vis * 2], smp[idx + num_vis * 3]], ["gen0", "gen1", "gen2", "gen3"])
                    all_imgs.append(img)
                img = np.concatenate(all_imgs, axis=1)
                writer.add_image('tr_vis/gen', torch.as_tensor(img), epoch)


    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'opt_lenc': self.opt_lenc.state_dict(),
            'opt_ldec': self.opt_ldec.state_dict(),
            'dec': self.dec.state_dict(),
            'enc': self.enc.state_dict(),
            'ldec': self.ldec.state_dict(),
            'lenc': self.lenc.state_dict(),
            'epoch': epoch,
            'step': step
        }
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)
       
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def generate(self, num_shapes=10, num_points=2048, save=False):
        with torch.no_grad():
            cate = self.cfg.data.cates[0]
            self.ldec.eval()
            self.dec.eval()
            import pdb; pdb.set_trace()
            z = self.ldec(torch.randn(num_shapes, 128).cuda())
            imgs = self.dec(z)
            if save:
                visualize_point_clouds_img(imgs.cpu().numpy(), None, cate=cate)
                os.makedirs("results", exist_ok=True)
                output_file = f"./results/{cate}_sample.npy"
                with open(output_file, "wb") as f:
                    np.save(f, imgs.cpu().numpy())
            return imgs
