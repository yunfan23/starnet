import importlib
import os

import evaluation.emd.emd_module as emd
import numpy as np
import torch
import torch.nn as nn
import tqdm
from evaluation.chamfer_distance import ChamferDistanceMean
from evaluation.evaluation_metrics import EMD_CD
from evaluation.StructuralLosses.nn_distance import nn_distance  # noqa

from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import (count_parameters, get_opt, get_prior,
                                  set_random_seed, visualize_point_clouds_3d)
from trainers.utils.vis_utils import (visualize_point_clouds_3d,
                                      visualize_point_clouds_img)


class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        self.loss_emd = emd.emdModule().cuda()
        self.loss_cd = ChamferDistanceMean().cuda()

        # The networks
        decoder_lib = importlib.import_module(cfg.models.decoder.type)
        self.decoder = decoder_lib.Decoder(cfg.models.decoder)
        self.decoder.cuda()
        print("Decoder:")
        print(self.decoder)
        print(f"Decoder : {count_parameters(self.decoder)}")

        encoder_lib = importlib.import_module(cfg.models.encoder.type)
        self.encoder = encoder_lib.Encoder(cfg.models.encoder)
        self.encoder.cuda()
        print("Encoder:")
        print(self.encoder)
        print(f"Encoder : {count_parameters(self.encoder)}")

        # The optimizer
        if not (hasattr(self.cfg.trainer, "opt_enc") and
                hasattr(self.cfg.trainer, "opt_dec")):
            self.cfg.trainer.opt_enc = self.cfg.trainer.opt
            self.cfg.trainer.opt_dec = self.cfg.trainer.opt

        self.opt_enc, self.scheduler_enc = get_opt(
            self.encoder.parameters(), self.cfg.trainer.opt_enc)
        self.opt_dec, self.scheduler_dec = get_opt(
            self.decoder.parameters(), self.cfg.trainer.opt_dec)

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
                writer.add_scalar('train/opt_dec_lr', self.scheduler_dec.get_lr()[0], epoch)
        if self.scheduler_enc is not None:
            self.scheduler_enc.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar('train/opt_enc_lr', self.scheduler_enc.get_lr()[0], epoch)
                
    def loss_fn(self, recon, gt, weight=1.):
        loss_cd = self.loss_cd(recon, gt).mean()
        dl, dr = nn_distance(recon, gt)
        loss_cd = dl.mean() + dr.mean()

        recon = (recon + 1) / 2.
        gt = (gt + 1.) / 2.
        loss_emd = self.loss_emd(recon, gt, eps=0.005, iters=50)[0]
        loss_emd = torch.sqrt(loss_emd).mean()

        loss = loss_emd + weight * loss_cd
        return loss, loss_cd, loss_emd


    def update(self, dataset, *args, **kwargs):
        total_loss, total_loss_cd, total_loss_emd = 0., 0., 0.
        avg_loss, avg_loss_cd, avg_loss_emd = 0., 0., 0.
        num_items = 0.
        loss = 0.
        for _, data in enumerate(dataset):
            if 'no_update' in kwargs:
                no_update = kwargs['no_update']
            else:
                no_update = False
            if not no_update:
                self.encoder.train()
                self.decoder.train()
                self.opt_enc.zero_grad()
                self.opt_dec.zero_grad()

            tr_pts = data['tr_points'].cuda()
            # import pdb; pdb.set_trace()
            bs = tr_pts.size(0)
            tr_pts.requires_grad_(True)
            z, _ = self.encoder(tr_pts)
            recon_pts = self.decoder(z)

            cd_weight = getattr(self.cfg.viz, "cd_weight", 25.)
            loss, loss_cd, loss_emd = self.loss_fn(recon_pts, tr_pts, cd_weight)

            if not no_update:
                loss.backward()
                self.opt_enc.step()
                self.opt_dec.step()

            total_loss += loss.detach().cpu().item() * bs
            num_items += bs
            avg_loss = total_loss / num_items

            total_loss_cd += loss_cd.detach().cpu().item() * bs
            avg_loss_cd = total_loss_cd / num_items

            total_loss_emd += loss_emd.detach().cpu().item() * bs
            avg_loss_emd = total_loss_emd / num_items

        return {
            'loss': avg_loss,
            'loss_cd': avg_loss_cd,
            'loss_emd': avg_loss_emd
        }

    def log_train(self, train_info, train_data, writer=None, step=None, epoch=None, visualize=False, **kwargs):
        if writer is None:
            return

        # Log training information to tensorboard
        train_info = {k: (v.cpu() if not isinstance(v, float) else v) for k, v in train_info.items()}
        for k, v in train_info.items():
            if not ('loss' in k):
                continue
            writer.add_scalar('train/' + k, v, epoch)

        if visualize:
            with torch.no_grad():
                print("Visualize: Epoch %s" % epoch)
                data = next(iter(train_data))
                inp = data['tr_points']
                num_vis = min(getattr(self.cfg.viz, "num_vis_samples", 5), inp.size(0))
                rec = self.reconstruct(inp[:num_vis].cuda(), num_points=inp.size(1))
                all_imgs = []
                for idx in range(num_vis):
                    img = visualize_point_clouds_3d([rec[idx], inp[idx]], ["recon", "gt"])
                    all_imgs.append(img)
                img = np.concatenate(all_imgs, axis=1)
                writer.add_image('tr_vis/recon', torch.as_tensor(img), epoch)


    def validate(self, data_loader, epoch, *args, **kwargs):
        print("Validation (reconstruction):")
        all_rec_denorm, all_inp_denorm = [], []
        for data in tqdm.tqdm(data_loader):
            # ref_pts = data['te_points'].cuda()
            inp_pts = data['tr_points'].cuda()
            m = data['mean'].cuda()
            std = data['std'].cuda()
            rec_pts = self.reconstruct(inp_pts, num_points=inp_pts.size(1))

            # denormalize
            inp_pts_denorm = inp_pts.clone() * std + m
            # ref_pts_denorm = ref_pts.clone() * std + m
            rec_pts_denorm = rec_pts * std + m

            # all_inp.append(inp_pts)
            all_inp_denorm.append(inp_pts_denorm.view(*inp_pts.size()))
            # all_ref_denorm.append(ref_pts_denorm.view(*ref_pts.size()))
            all_rec_denorm.append(rec_pts_denorm.view(*rec_pts.size()))
            # all_ref_denorm.append(ref_pts_denorm)

        # inp = torch.cat(all_inp, dim=0)
        # rec = torch.cat(all_rec, dim=0)
        # ref_denorm = torch.cat(all_ref_denorm, dim=0)
        inp_denorm = torch.cat(all_inp_denorm, dim=0)
        rec_denorm = torch.cat(all_rec_denorm, dim=0)

        # Oracle CD/EMD, will compute only once
        # if self.oracle_res is None:
        #     rec_res = EMD_CD(inp_denorm, ref_denorm, 1)
        #     rec_res = {
        #         ("val/rec/%s" % k): (v if isinstance(v, float) else v.item())
        #         for k, v in rec_res.items()}
        #     all_res.update(rec_res)
        #     print("Validation oracle (denormalize) Epoch:%d " % epoch, rec_res)
        #     self.oracle_res = rec_res
        # else:
        #     all_res.update(self.oracle_res)

        # Reconstruction CD/EMD
        all_res = {}
        val_bs = getattr(self.cfg.trainer, "val_metrics_batch_size", 100)
        rec_res = EMD_CD(rec_denorm, inp_denorm, val_bs)
        rec_res = {("val/rec/%s" % k): (v if isinstance(v, float) else v.item()) for k, v in rec_res.items()}
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

    def generate(self, dataloader):
        print(self.cfg.data.cates[0])
        cate = self.cfg.data.cates[0]
        for idx, data in enumerate(dataloader):
            with torch.no_grad():
                self.encoder.eval()
                self.decoder.eval()
                inps = data["tr_points"].cuda()
                z, _ = self.encoder(inps)
                samples = self.decoder(z)
                # import pdb; pdb.set_trace()
                os.makedirs("results", exist_ok=True)
                output_filename = f"./results/{cate}_batch_{idx}.npy"
                with open(output_filename, 'wb') as f:
                    np.save(f, samples.cpu().numpy())
                visualize_point_clouds_img(samples.cpu().numpy(), inps.cpu().numpy(), bs=idx, cate=cate)

    def reconstruct(self, inps, num_points=2048):
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            z, _ = self.encoder(inps)
            samples = self.decoder(z)
            return samples
