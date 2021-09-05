import importlib
import math
import os
import pdb
import random

import numpy as np
import torch
import tqdm
from evaluation.evaluation_metrics import (compute_all_metrics,
                                           jsd_between_point_cloud_sets)
from Frechet.FPD import calculate_fpd

from trainers.ae_sparenet_trainer_3D import Trainer as BaseTrainer
from trainers.utils.gan_losses import dis_loss, gen_loss, gradient_penalty
from trainers.utils.utils import (count_parameters, get_opt,
                                  normalize_point_clouds)
from trainers.utils.vis_utils import (visualize_point_clouds_3d,
                                      visualize_point_clouds_img)


class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)

        # Now initialize the GAN part
        gen_lib = importlib.import_module(cfg.models.gen.type)
        self.gen = gen_lib.Generator(cfg, cfg.models.gen)
        self.gen.cuda()
        print(f"Generator : {count_parameters(self.gen)}")
        print(self.gen)

        dis_lib = importlib.import_module(cfg.models.dis.type)
        self.dis = dis_lib.Discriminator(cfg, cfg.models.dis)
        self.dis.cuda()
        print(f"Discriminator : {count_parameters(self.dis)}")
        print(self.dis)

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
        if self.scheduler_dis is not None:
            self.scheduler_dis.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar('train/opt_dis_lr', self.scheduler_dis.get_lr()[0], epoch)

        if self.scheduler_gen is not None:
            self.scheduler_gen.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar('train/opt_gen_lr', self.scheduler_gen.get_lr()[0], epoch)

    def _update_gan_(self, data, gen=False):
        self.encoder.eval()
        self.gen.train()
        self.dis.train()
        self.opt_gen.zero_grad()
        self.opt_dis.zero_grad()

        tr_pts = data['tr_points'].cuda()  # (B, #points, 3)smn_ae_trainer.py
        batch_size = tr_pts.size(0)
        with torch.no_grad():
            x_real, _ = self.encoder(tr_pts)

        x_real.requires_grad = True
        x_fake = self.gen(bs=batch_size)

        x_inp = torch.cat([x_real, x_fake], dim=0)
        d_res = self.dis(x_inp, return_all=True)
        d_out = d_res['x']
        d_real = d_out[:batch_size, ...]
        d_fake = d_out[batch_size:, ...]

        loss_res = {}
        loss_type = getattr(self.cfg.trainer, "gan_loss_type", "wgan")
        if gen:
            gen_loss_weight = getattr(self.cfg.trainer, "gen_loss_weight", 1.)
            loss_gen, gen_loss_res = gen_loss(d_real, d_fake, weight=gen_loss_weight, loss_type=loss_type)
            loss_gen.backward()
            self.opt_gen.step()
            loss_res.update({("train/gan_pass/gen/%s" % k): v for k, v in gen_loss_res.items() })
            # loss_res['loss'] = loss.detach().cpu().item()
        else:
            # Get gradient penalty
            gp_weight = getattr(self.cfg.trainer, 'gp_weight', 10.)
            gp_type = getattr(self.cfg.trainer, 'gp_type', "zero_center")
            gp, gp_res = gradient_penalty(x_real, x_fake, d_real, d_fake, weight=gp_weight, gp_type=gp_type)
            loss_res.update({("train/gan_pass/gp_loss/%s" % k): v for k, v in gp_res.items() })
            dis_loss_weight = getattr(self.cfg.trainer, "dis_loss_weight", 1.)
            loss_dis, dis_loss_res = dis_loss(d_real, d_fake, weight=dis_loss_weight, loss_type=loss_type)
            loss_dis = loss_dis + gp
            loss_dis.backward()
            self.opt_dis.step()
            loss_res.update({("train/gan_pass/dis/%s" % k): v for k, v in dis_loss_res.items()})

        # loss_res['x_real'] = x_real.clone().detach().cpu()
        # loss_res['x_fake'] = x_fake.clone().detach().cpu()
        return loss_res

    def update(self, dataset):
        res = {}
        for _, data in enumerate(dataset):
            self.total_gan_iters += 1
            if self.total_gan_iters % self.n_critics == 0:
                gen_res = self._update_gan_(data, gen=True)
                res.update(gen_res)
            dis_res = self._update_gan_(data, gen=False)
            res.update(dis_res)
        return res

    def log_train(self, train_info, train_data, writer=None, step=None, epoch=None, visualize=False, **kwargs):
        train_info = {k: (v.cpu() if not isinstance(v, float) else v) for k, v in train_info.items()}
        for k, v in train_info.items():
            if not ('loss' in k):
                continue
            writer.add_scalar('train/' + k, v, epoch)
        # writer.add_histogram('tr/latent_real', train_info['x_real'], epoch)
        # writer.add_histogram('tr/latent_fake', train_info['x_fake'], epoch)

        if visualize:
            with torch.no_grad():
                print("Visualize generation results: %s" % epoch)
                # data = next(iter(train_data))
                # gtr = data['te_points']  # ground truth point cloud
                num_vis = getattr(self.cfg.viz, "num_vis_samples", 5)
                smp = self.sample(num_shapes=num_vis * 4)

                all_imgs = []
                for idx in range(num_vis):
                    # img = visualize_point_clouds_3d([smp[idx], gtr[idx]], ["gen", "ref"])
                    img = visualize_point_clouds_3d([smp[idx], smp[idx + num_vis], smp[idx + num_vis * 2], smp[idx + num_vis * 3]], ["gen0", "gen1", "gen2", "gen3"])
                    all_imgs.append(img)
                img = np.concatenate(all_imgs, axis=1)
                writer.add_image('tr_vis/gen', torch.as_tensor(img), epoch)


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

    def sample(self, num_shapes=10):
        with torch.no_grad():
            self.gen.eval()
            self.decoder.eval()
            z = self.gen(bs=num_shapes)
            pcs = self.decoder(z)
            return pcs

    def generate(self, num_shapes=10, num_points=2048):
        with torch.no_grad():
            cate = self.cfg.data.cates[0]
            self.gen.eval()
            self.decoder.eval()
            z = self.gen(bs=num_shapes)
            imgs = self.decoder(z)
            visualize_point_clouds_img(imgs.cpu().numpy(), None, cate=cate)
            os.makedirs("results", exist_ok=True)
            output_file = f"./results/{cate}_sample.npy"
            with open(output_file, "wb") as f:
                np.save(f, imgs.cpu().numpy())

    def validate(self, test_loader, evaluation=False, *args, **kwargs):
        all_res = {}
        max_gen_vali_shape = int(getattr(self.cfg.trainer, "max_gen_validate_shapes", 100))

        # calculate fpd score
        # fpd = self.validate_fpd(test_loader)
        # all_res = {"FPD": fpd}

        all_ref, all_smp = [], []
        # reference point clouds
        for data in tqdm.tqdm(test_loader, 'Dataset'):
            ref_pts = data['tr_points']
            all_ref.append(ref_pts)
        ref = torch.cat(all_ref, dim=0).cuda()

        ref_num = ref.size(0)
        for i in tqdm.tqdm(range(0, math.ceil(ref_num / max_gen_vali_shape)), 'Generate'):
            with torch.no_grad():
                self.gen.eval()
                self.decoder.eval()
                z = torch.randn((max_gen_vali_shape, 128)).cuda()
                w = self.gen(z=z)
                pcs = self.decoder(w)
                all_smp.append(pcs)
        smp = torch.cat(all_smp, dim=0)[:ref_num]
        np.save(os.path.join(self.cfg.save_dir, 'val', 'smp.npy'), smp.detach().cpu().numpy())
        smp = normalize_point_clouds(smp)
        ref = normalize_point_clouds(ref)
        # sub_sampled = random.sample(range(ref.size(0)), 200)
        # smp = smp[sub_sampled, ...].contiguous()
        # ref = ref[sub_sampled, ...].contiguous()
        print(f"Sample Shape {smp.shape}")
        print(f"Reference Shape {ref.shape}")

        jsd = jsd_between_point_cloud_sets(smp.cpu().numpy(), ref.cpu().numpy())
        all_res.update({"JSD": jsd})
        print(f"JSD: {jsd}")

        # pdb.set_trace()
        if evaluation:
            gen_res = compute_all_metrics(smp, ref, batch_size=int(getattr(self.cfg.trainer, "val_metrics_batch_size", 100)), accelerated_cd=True)
            all_res.update({("val/gen/%s" % k): (v if isinstance(v, float) else v.item()) for k, v in gen_res.items()})
        print("Validation Done")

        return all_res


    def validate_fpd(self, dataloader, *args, **kwargs):
        with torch.no_grad():
            self.gen.eval()
            self.decoder.eval()
            all_ref, all_smp = [], []
            for _ in tqdm.tqdm(range(50), desc="Sample"):
                z = torch.randn((100, 128)).cuda()
                w = self.gen(z=z)
                pcs = self.decoder(w)
                all_smp.append(pcs)

            for data in tqdm.tqdm(dataloader, desc="Data"):
                inp_pts = data['tr_points'].cuda()
                all_ref.append(inp_pts)

            smp = torch.cat(all_smp, dim=0)
            ref = torch.cat(all_ref, dim=0)
            # import pdb; pdb.set_trace()
            # smp = (smp) / (smp.max(dim=1, keepdim=True)[0] - smp.min(dim=1, keepdim=True)[0]) / 2
            # ref = (ref) / (ref.max(dim=1, keepdim=True)[0] - ref.min(dim=1, keepdim=True)[0]) / 2
            # smp = pc_normalize(smp)
            # ref = pc_normalize(ref)
            # smp = (smp - smp.min(dim=1, keepdim=True)[0]) / (smp.max(dim=1, keepdim=True)[0] - smp.min(dim=1, keepdim=True)[0]) / 2
            # ref = (ref - ref.min(dim=1, keepdim=True)[0]) / (ref.max(dim=1, keepdim=True)[0] - ref.min(dim=1, keepdim=True)[0]) / 2
            # smp = (smp) / (smp.max() - smp.min()) / 2
            # ref = (ref) / (ref.max() - ref.min()) / 2
            # smp = (smp - smp.min()) / (smp.max() - smp.min()) / 4
            # ref = (ref - ref.min()) / (ref.max() - ref.min()) / 4
            fpd = calculate_fpd(smp, ref, batch_size=100, dims=1808)
            # print(f'Frechet Pointcloud Distance {fpd:.3f}')
            return fpd
