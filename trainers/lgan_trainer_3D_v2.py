import importlib
import os
import pdb
import random

import numpy as np
import torch
import torch.nn as nn
import tqdm

from trainers.ae_sparenet_trainer_3D import Trainer as BaseTrainer
from trainers.utils.gan_losses import dis_loss, gen_loss, gradient_penalty
from trainers.utils.utils import count_parameters, get_opt
from trainers.utils.vis_utils import (visualize_point_clouds_3d,
                                      visualize_point_clouds_img)

try:
    from evaluation.evaluation_metrics import compute_all_metrics

    eval_generation = True
except:  # noqa
    eval_generation = False


class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)

        # Now initialize the GAN part
        gen_lib = importlib.import_module(cfg.models.gen.type)
        self.gen = gen_lib.Generator(cfg, cfg.models.gen)
        # self.gen = nn.DataParallel(self.gen)
        self.gen.cuda()
        print("Generator:")
        print(self.gen)

        dis_lib = importlib.import_module(cfg.models.dis.type)
        self.dis = dis_lib.Discriminator(cfg, cfg.models.dis)
        # self.dis = nn.DataParallel(self.dis)
        self.dis.cuda()
        print("Discriminator:")
        print(self.dis)

        self.dis2 = Discriminator(2048)
        self.dis2.cuda()
        print("Discriminator v2:")
        print(self.dis2)

        # Optimizers
        if not (hasattr(self.cfg.trainer, "opt_gen") and
                hasattr(self.cfg.trainer, "opt_dis")):
            self.cfg.trainer.opt_gen = self.cfg.trainer.opt
            self.cfg.trainer.opt_dis = self.cfg.trainer.opt
        self.opt_gen, self.scheduler_gen = get_opt(
            self.gen.parameters(), self.cfg.trainer.opt_gen)
        self.opt_dis, self.scheduler_dis = get_opt(
            self.dis.parameters(), self.cfg.trainer.opt_dis)

        self.opt_dis2, self.scheduler_dis2 = get_opt(
            self.dis2.parameters(), self.cfg.trainer.opt_dis)

        # book keeping
        self.total_iters = 0
        self.total_gan_iters = 0
        self.n_critics = getattr(self.cfg.trainer, "n_critics", 1)
        self.gan_only = getattr(self.cfg.trainer, "gan_only", True)

        # If pretrained AE, then load it up
        if hasattr(self.cfg.trainer, "ae_pretrained"):
            ckpt = torch.load(self.cfg.trainer.ae_pretrained)
            print(self.cfg.trainer.ae_pretrained)
            strict = getattr(self.cfg.trainer, "resume_strict", True)
            self.encoder.load_state_dict(ckpt['enc'], strict=strict)
            self.decoder.load_state_dict(ckpt['dec'], strict=strict)
            if getattr(self.cfg.trainer, "resume_opt", False):
                self.opt_enc.load_state_dict(ckpt['opt_enc'])
                self.opt_dec.load_state_dict(ckpt['opt_dec'])
        self.gan_pass_update_enc = getattr(
            self.cfg.trainer, "gan_pass_update_enc", False)

    def epoch_end(self, epoch, writer=None, **kwargs):
        super().epoch_end(epoch, writer=writer)

        if self.scheduler_dis is not None:
            self.scheduler_dis.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_dis_lr', self.scheduler_dis.get_lr()[0], epoch)

        if self.scheduler_gen is not None:
            self.scheduler_gen.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_gen_lr', self.scheduler_gen.get_lr()[0], epoch)

    def _update_gan_(self, data, gen=False):
        self.gen.train()
        self.dis.train()
        self.opt_gen.zero_grad()
        self.opt_dis.zero_grad()
        if self.gan_pass_update_enc:
            self.encoder.train()
            self.opt_enc.zero_grad()
        else:
            self.encoder.eval()
            self.decoder.eval()

        tr_pts = data['tr_points'].cuda()  # (B, #points, 3)smn_ae_trainer.py
        x_real, _ = self.encoder(tr_pts)
        batch_size = x_real.size(0)
        x_fake = self.gen(bs=batch_size)

        x_inp = torch.cat([x_real, x_fake], dim=0)
        d_res = self.dis(x_inp, return_all=True)
        d_out = d_res['x']
        d_real = d_out[:batch_size, ...]
        d_fake = d_out[batch_size:, ...]

        pc_inp = self.decoder(x_inp)[0]
        pc_real = pc_inp[:batch_size, ...]
        pc_fake = pc_inp[batch_size:, ...]

        d2_out = self.dis2(pc_inp)
        d2_real = d2_out[:batch_size, ...]
        d2_fake = d2_out[batch_size:, ...]

        loss_res = {}
        loss_type = getattr(self.cfg.trainer, "gan_loss_type", "wgan")
        if gen:
            gen_loss_weight = getattr(self.cfg.trainer, "gen_loss_weight", 1.)
            loss_gen2, _ = gen_loss(d2_real, d2_fake, weight=gen_loss_weight, loss_type=loss_type)
            loss_gen, gen_loss_res = gen_loss(d_real, d_fake, weight=gen_loss_weight, loss_type=loss_type)
            loss_gen = loss_gen + loss_gen2
            loss_gen.backward()
            self.opt_gen.step()
            if self.gan_pass_update_enc:
                assert self.opt_enc is not None
                self.opt_enc.step()
            loss_res.update({("train/gan_pass/gen/%s" % k): v for k, v in gen_loss_res.items()})
        else:
            # Get gradient penalty
            gp_weight = getattr(self.cfg.trainer, 'gp_weight', 0.)
            dis_loss_weight = getattr(self.cfg.trainer, "dis_loss_weight", 1.)
            gp_type = getattr(self.cfg.trainer, 'gp_type', "zero_center")

            gp, gp_res = gradient_penalty(x_real, x_fake, d_real, d_fake, weight=gp_weight, gp_type=gp_type)
            # gp2, gp2_res = gradient_penalty(pc_real, pc_fake, d2_real, d2_fake, weight=gp_weight, gp_type=gp_type)
            gp2 = 0
            loss_dis, dis_loss_res = dis_loss(d_real, d_fake, weight=dis_loss_weight, loss_type=loss_type)
            loss_dis2, dis2_loss_res = dis_loss(d2_real, d2_fake, weight=dis_loss_weight, loss_type=loss_type)
            loss_dis_all = loss_dis + loss_dis2 + gp2
            loss_dis_all.backward()
            self.opt_dis2.step()
            self.opt_dis.step()

            loss_res.update({("train/gan_pass/gp_loss/%s" % k): v for k, v in gp_res.items() })
            # loss_res.update({("train/gan_pass/gp2_loss/%s" % k): v for k, v in gp2_res.items() })
            loss_res.update({("train/gan_pass/dis/%s" % k): v for k, v in dis_loss_res.items()})
            loss_res.update({("train/gan_pass/dis2/%s" % k): v for k, v in dis2_loss_res.items()})


        loss_res['x_real'] = x_real.clone().detach().cpu()
        loss_res['x_fake'] = x_fake.clone().detach().cpu()
        return loss_res

    def update_lgan(self, data):
        self.total_gan_iters += 1
        res = {}
        if self.total_gan_iters % self.n_critics == 0:
            gen_res = self._update_gan_(data, gen=True)
            res.update(gen_res)
        dis_res = self._update_gan_(data, gen=False)
        res.update(dis_res)
        return res

    def update(self, data, *args, **kwargs):
        res = {}
        if not self.gan_only:
            ae_res = super().update(data, *args, **kwargs)
            res.update(ae_res)
        gan_res = self.update_lgan(data)
        res.update(gan_res)
        return res

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        with torch.no_grad():
            train_info.update(super().update(train_data, no_update=True))
        super().log_train(train_info, train_data, writer=writer, step=step,
                          epoch=epoch, visualize=visualize)
        if step is not None:
            writer.add_histogram('tr/latent_real', train_info['x_real'], step)
            writer.add_histogram('tr/latent_fake', train_info['x_fake'], step)
        else:
            assert epoch is not None
            writer.add_histogram('tr/latent_real', train_info['x_real'], epoch)
            writer.add_histogram('tr/latent_fake', train_info['x_fake'], epoch)

        if visualize:
            with torch.no_grad():
                print("Visualize generation results: %s" % step)
                gtr = train_data['te_points']  # ground truth point cloud
                inp = train_data['tr_points']  # input for encoder
                num_vis = min(
                    getattr(self.cfg.viz, "num_vis_samples", 5),
                    gtr.size(0)
                )
                smp = self.sample(num_shapes=num_vis, num_points=inp.size(1))

                all_imgs = []
                for idx in range(num_vis):
                    img = visualize_point_clouds_3d(
                        [smp[idx], gtr[idx]], ["gen", "ref"])
                    all_imgs.append(img)
                img = np.concatenate(all_imgs, axis=1)
                writer.add_image('tr_vis/gen', torch.as_tensor(img), step)


    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'opt_enc': self.opt_enc.state_dict(),
            'opt_dec': self.opt_dec.state_dict(),
            'opt_dis': self.opt_dis.state_dict(),
            'opt_gen': self.opt_gen.state_dict(),
            'dec': self.decoder.state_dict(),
            'enc': self.encoder.state_dict(),
            'dis': self.dis.state_dict(),
            'gen': self.gen.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)

    def resume(self, path, strict=True, **args):
        ckpt = torch.load(path)
        self.encoder.load_state_dict(ckpt['enc'], strict=strict)
        self.decoder.load_state_dict(ckpt['dec'], strict=strict)
        self.opt_enc.load_state_dict(ckpt['opt_enc'])
        self.opt_dec.load_state_dict(ckpt['opt_dec'])
        start_epoch = ckpt['epoch']

        if 'gen' in ckpt:
            self.gen.load_state_dict(ckpt['gen'], strict=strict)
        if 'dis' in ckpt:
            self.dis.load_state_dict(ckpt['dis'], strict=strict)
        if 'opt_gen' in ckpt:
            self.opt_gen.load_state_dict(ckpt['opt_gen'])
        if 'opt_dis' in ckpt:
            self.opt_dis.load_state_dict(ckpt['opt_dis'])
        return start_epoch


    def sample(self, num_shapes=10, num_points=2048):
        with torch.no_grad():
            cate = self.cfg.data.cates[0]
            self.gen.eval()
            self.decoder.eval()
            z = self.gen(bs=num_shapes)
            imgs, _ = self.decoder(z)
            # pdb.set_trace()
            visualize_point_clouds_img(imgs.cpu().numpy(), None, cate=cate)
            output_file = f"{cate}_sample.npy"
            with open(output_file, "wb") as f:
                np.save(f, imgs.cpu().numpy())
            return imgs

    def validate(self, test_loader, epoch, *args, **kwargs):
        all_res = {}

        if eval_generation:
            with torch.no_grad():
                print("l-GAN validation:")
                all_ref, all_smp = [], []
                for data in tqdm.tqdm(test_loader):
                    ref_pts = data['te_points'].cuda()
                    inp_pts = data['tr_points'].cuda()
                    smp_pts = self.sample(
                        num_shapes=inp_pts.size(0),
                        num_points=inp_pts.size(1),
                    )
                    all_smp.append(smp_pts.view(
                        ref_pts.size(0), ref_pts.size(1), ref_pts.size(2)))
                    all_ref.append(
                        ref_pts.view(ref_pts.size(0), ref_pts.size(1),
                                     ref_pts.size(2)))

                smp = torch.cat(all_smp, dim=0)
                np.save(
                    os.path.join(self.cfg.save_dir, 'val',
                                 'smp_ep%d.npy' % epoch),
                    smp.detach().cpu().numpy()
                )
                ref = torch.cat(all_ref, dim=0)

                # Sample CD/EMD
                # step 1: subsample shapes
                max_gen_vali_shape = int(getattr(
                    self.cfg.trainer, "max_gen_validate_shapes",
                    int(smp.size(0))))
                sub_sampled = random.sample(
                    range(smp.size(0)), min(smp.size(0), max_gen_vali_shape))
                smp_sub = smp[sub_sampled, ...].contiguous()
                ref_sub = ref[sub_sampled, ...].contiguous()

                gen_res = compute_all_metrics(
                    smp_sub, ref_sub,
                    batch_size=int(getattr(
                        self.cfg.trainer, "val_metrics_batch_size", 100)),
                    accelerated_cd=True
                )
                all_res = {
                    ("val/gen/%s" % k):
                        (v if isinstance(v, float) else v.item())
                    for k, v in gen_res.items()}
                print("Validation Sample (unit) Epoch:%d " % epoch, gen_res)


        # Call super class validation
        if getattr(self.cfg.trainer, "validate_recon", False):
            all_res.update(super().validate(
                test_loader, epoch, *args, **kwargs))

        return all_res


class Discriminator(nn.Module):
    def __init__(self, num_pts):
        super(Discriminator, self).__init__()
        self.num_pts = num_pts
        self.fc1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(256, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool1d(self.num_pts, 1)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batchsize = x.size()[0]
        x = x.transpose(1, 2)
        x1 = self.fc1(x)
        x2 = self.maxpool(x1)
        x2 = x2.view(batchsize, 1024)
        x3 = self.mlp(x2)

        return x3
