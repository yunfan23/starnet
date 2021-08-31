import importlib
import os
import random

import numpy as np
import torch
import torch.nn as nn
import tqdm
from models.generators.mlp_gen import truncated_normal

from trainers.base_trainer import BaseTrainer
from trainers.utils.gan_losses import dis_loss, gen_loss, gradient_penalty
from trainers.utils.utils import ForkedPdb, get_opt, visualize_point_clouds_3d

try:
    from evaluation.evaluation_metrics import compute_all_metrics

    eval_generation = True
except:  # noqa
    eval_generation = False


class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args

        # Now initialize the GAN part
        gen_lib = importlib.import_module(cfg.models.gen.type)
        self.gen = gen_lib.Generator(cfg.models.gen)
        self.gen.cuda()
        self.gen = nn.DataParallel(self.gen)

        print("Generator:")
        print(self.gen)

        dis_lib = importlib.import_module(cfg.models.dis.type)
        self.dis = dis_lib.Discriminator(cfg.models.dis)
        self.dis.cuda()
        self.dis = nn.DataParallel(self.dis)

        print("Discriminator:")
        print(self.dis)

        # import pdb; pdb.set_trace()
        # Optimizers
        if not (hasattr(self.cfg.trainer, "opt_gen") and
                hasattr(self.cfg.trainer, "opt_dis")):
            self.cfg.trainer.opt_gen = self.cfg.trainer.opt
            self.cfg.trainer.opt_dis = self.cfg.trainer.opt
        self.opt_gen, self.scheduler_gen = get_opt(
            self.gen.parameters(), self.cfg.trainer.opt_gen)
        self.opt_dis, self.scheduler_dis = get_opt(
            self.dis.parameters(), self.cfg.trainer.opt_dis)

        # book keeping
        self.total_iters = 0
        self.total_gan_iters = 0
        self.n_critics = getattr(self.cfg.trainer, "n_critics", 1)
        self.gan_only = getattr(self.cfg.trainer, "gan_only", True)
        self.prior_type = getattr(cfg.models, "prior", "gaussian")
        # get prior noise when initializing
        self.num_vis = getattr(self.cfg.inference, "num_vis_samples", 5)
        self.same_prior = getattr(self.cfg.inference, "same_prior", False)
        if self.same_prior:
            self.noise = self.get_prior(bs=self.num_vis).cuda()
        else:
            self.noise = None

    def multi_gpu_wrapper(self, wrapper):
        self.gen = wrapper(self.gen)
        self.dis = wrapper(self.dis)

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

        x_real = data['tr_points'].cuda()
        x_real.requires_grad_(True)
        batch_size = x_real.size(0)
        z = self.get_prior(batch_size).cuda()
        z.requires_grad_(True)
        x_fake = self.gen(z)
        x_fake = x_fake.transpose(1, 2)
        x_inp = torch.cat([x_real, x_fake], dim=0)
        d_res = self.dis(x_inp)
        d_real = d_res[:batch_size, ...]
        d_fake = d_res[batch_size:, ...]

        loss_res = {}
        loss_type = getattr(self.cfg.trainer, "gan_loss_type", "wgan")
        if gen:
            gen_loss_weight = getattr(self.cfg.trainer, "gen_loss_weight", 1.)
            loss_gen, gen_loss_res = gen_loss(d_real, d_fake, weight=gen_loss_weight, loss_type=loss_type)
            loss_gen.backward()
            self.opt_gen.step()
            loss_res.update({("train/gan_pass/gen/%s" % k): v for k, v in gen_loss_res.items()})
        else:
            # Get gradient penalty
            gp_weight = self.cfg.trainer.gp_weight
            gp_type = getattr(self.cfg.trainer, 'gp_type', "zero_center")
            gp, gp_res = gradient_penalty(x_real, x_fake, d_real, d_fake, weight=gp_weight, gp_type=gp_type)
            loss_res.update({("train/gan_pass/gp_loss/%s" % k): v for k, v in gp_res.items()})
            dis_loss_weight = getattr(self.cfg.trainer, "dis_loss_weight", 1.)
            loss_dis, dis_loss_res = dis_loss(d_real, d_fake, weight=dis_loss_weight, loss_type=loss_type)
            loss_dis += gp
            loss_dis.backward()
            self.opt_dis.step()
            loss_res.update({("train/gan_pass/dis/%s" % k): v for k, v in dis_loss_res.items()})

        loss_res['x_real'] = x_real.clone().detach().cpu()
        loss_res['x_fake'] = x_fake.clone().detach().cpu()
        return loss_res

    def update_gan(self, data):
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
        gan_res = self.update_gan(data)
        res.update(gan_res)
        return res

    def log_train(self, train_info, train_data, writer=None, step=None, epoch=None, visualize=False, **kwargs):
        # Log training information to tensorboard
        # ForkedPdb().set_trace()
        train_info = {k: (v.cpu() if not isinstance(v, float) else v) for k, v in train_info.items()}
        for k, v in train_info.items():
            if not ('loss' in k):
                continue
            assert epoch is not None
            writer.add_scalar('train/' + k, v, epoch)
        writer.add_histogram('tr/latent_real', train_info['x_real'], epoch)
        writer.add_histogram('tr/latent_fake', train_info['x_fake'], epoch)

        if visualize:
            with torch.no_grad():
                print("Visualize generation results: %s" % step)
                gtr = train_data['te_points']  # ground truth point cloud
                # inp = train_data['tr_points']  # input for encoder
                imgs = self.gen(self.get_prior(10).cuda())
                imgs = imgs.transpose(1, 2)
                all_imgs = []
                for idx in range(self.num_vis):
                    img = visualize_point_clouds_3d([imgs[idx], gtr[idx]], ["gen", "ref"])
                    all_imgs.append(img)
                img = np.concatenate(all_imgs, axis=1)
                writer.add_image('tr_vis/gen', torch.as_tensor(img), epoch)

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'opt_dis': self.opt_dis.state_dict(),
            'opt_gen': self.opt_gen.state_dict(),
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

    def validate(self, test_loader, epoch, *args, **kwargs):
        all_res = {}

        if eval_generation:
            with torch.no_grad():
                print("Style-GAN validation:")
                all_ref, all_smp = [], []
                for data in tqdm.tqdm(test_loader):
                    ref_pts = data['te_points'].cuda()
                    inp_pts = data['tr_points'].cuda()
                    with torch.no_grad():
                        self.gen.eval()
                        # put noise in cuda device
                        outputs = self.gen(self.get_prior(ref_pts.size(0)).cuda())
                        smp_pts = outputs.transpose(1, 2).contiguous()
                    all_smp.append(smp_pts.view(ref_pts.size(0), ref_pts.size(1), ref_pts.size(2)))
                    all_ref.append(ref_pts.view(ref_pts.size(0), ref_pts.size(1), ref_pts.size(2)))

                smp = torch.cat(all_smp, dim=0)
                np.save(os.path.join(self.cfg.save_dir, 'val','smp_ep%d.npy' % epoch), smp.detach().cpu().numpy())
                ref = torch.cat(all_ref, dim=0)

                # Sample CD/EMD
                # step 1: subsample shapes
                max_gen_vali_shape = int(getattr(self.cfg.trainer, "max_gen_validate_shapes", int(smp.size(0))))
                sub_sampled = random.sample(range(smp.size(0)), min(smp.size(0), max_gen_vali_shape))
                smp_sub = smp[sub_sampled, ...].contiguous()
                ref_sub = ref[sub_sampled, ...].contiguous()

                gen_res = compute_all_metrics(
                    smp_sub, ref_sub,
                    batch_size=int(getattr(self.cfg.trainer, "val_metrics_batch_size", 100)), accelerated_cd=True)
                all_res = {("val/gen/%s" % k): (v if isinstance(v, float) else v.item())
                    for k, v in gen_res.items()}
                print("Validation Sample (unit) Epoch:%d " % epoch, gen_res)

        # Call super class validation
        if getattr(self.cfg.trainer, "validate_recon", False):
            all_res.update(super().validate(
                test_loader, epoch, *args, **kwargs))

        return all_res

    # get and save the same prior for shape generation
    def get_prior(self, bs, dim=128):
        truncate_std = getattr(self.cfg.models, "truncate_std", 2.)
        gaussian_scale = getattr(self.cfg.models, "gaussian_scale", 1.)
        if self.prior_type == "truncate_gaussian":
            noise = (torch.randn(bs, self.cfg.models.gen.z_dim) * gaussian_scale).cuda()
            noise = truncated_normal(noise, 0, gaussian_scale, truncate_std)
            return noise
        elif self.prior_type == "gaussian":
            return torch.randn(bs, self.cfg.models.gen.z_dim) * gaussian_scale
        else:
            raise NotImplementedError("Invalid prior type:%s" % self.prior_type)
