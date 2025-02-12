import importlib
import os
import random
import math
import pdb
import numpy as np
import torch
import torch.nn as nn
import tqdm
from models.generators.mlp_gen import truncated_normal

from trainers.base_trainer import BaseTrainer
from trainers.utils.gan_losses import dis_loss, gen_loss, gradient_penalty
from trainers.utils.utils import ForkedPdb, get_opt, visualize_point_clouds_3d, normalize_point_clouds
from evaluation.evaluation_metrics import compute_all_metrics, jsd_between_point_cloud_sets

try:
    from evaluation.evaluation_metrics import compute_all_metrics

    eval_generation = True
except:  # noqa
    eval_generation = False


class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        # super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args

        # Now initialize the GAN part
        gen_lib = importlib.import_module(cfg.models.gen.type)
        self.gen = gen_lib.Generator(cfg, cfg.models.gen)
        self.gen.cuda()
        # self.gen = nn.DataParallel(self.gen)
        print("Generator:")
        print(self.gen)

        dec_lib = importlib.import_module(cfg.models.decoder.type)
        self.dec = dec_lib.Decoder(cfg.models.decoder)
        self.dec.cuda()
        # self.dec = nn.DataParallel(self.dec)
        print("Decoder:")
        print(self.dec)

        dis_lib = importlib.import_module(cfg.models.dis.type)
        self.dis = dis_lib.Discriminator(cfg.models.dis)
        self.dis.cuda()
        # self.dis = nn.DataParallel(self.dis)
        print("Discriminator:")
        print(self.dis)

        # import pdb; pdb.set_trace()
        # Optimizers
        self.opt_dec, self.scheduler_dec = get_opt(self.dec.parameters(), self.cfg.trainer.opt_dec)
        self.opt_gen, self.scheduler_gen = get_opt(self.gen.parameters(), self.cfg.trainer.opt_gen)
        self.opt_dis, self.scheduler_dis = get_opt(self.dis.parameters(), self.cfg.trainer.opt_dis)

        # book keeping
        self.total_iters = 0
        self.total_gan_iters = 0
        self.n_critics = getattr(self.cfg.trainer, "n_critics", 1)

    def multi_gpu_wrapper(self, wrapper):
        self.dec = wrapper(self.dec)
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

        if self.scheduler_dec is not None:
            self.scheduler_dec.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_dec_lr', self.scheduler_dec.get_lr()[0], epoch)

    def _update_gan_(self, data, gen=False):
        self.dec.train()
        self.gen.train()
        self.dis.train()
        self.opt_dec.zero_grad()
        self.opt_gen.zero_grad()
        self.opt_dis.zero_grad()

        x_real = data['tr_points'].cuda()
        x_real.requires_grad_(True)
        batch_size = x_real.size(0)
        z = self.get_prior(batch_size).cuda()
        z.requires_grad_(True)
        x_fake = self.dec(self.gen(z))
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
            # self.opt_dec.step()
            self.opt_gen.step()
            loss_res.update({("train/gen/%s" % k): v for k, v in gen_loss_res.items()})
        else:
            # Get gradient penalty
            gp_weight = self.cfg.trainer.gp_weight
            gp_type = getattr(self.cfg.trainer, 'gp_type', "zero_center")
            gp, gp_res = gradient_penalty(x_real, x_fake, d_real, d_fake, weight=gp_weight, gp_type=gp_type)
            loss_res.update({("train/gp_loss/%s" % k): v for k, v in gp_res.items()})
            dis_loss_weight = getattr(self.cfg.trainer, "dis_loss_weight", 1.)
            loss_dis, dis_loss_res = dis_loss(d_real, d_fake, weight=dis_loss_weight, loss_type=loss_type)
            loss_dis += gp
            loss_dis.backward()
            self.opt_dis.step()
            loss_res.update({("train/dis/%s" % k): v for k, v in dis_loss_res.items()})

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

    def update(self, dataset, *args, **kwargs):
        for data in dataset:
            res = self.update_gan(data)
        return res

    def log_train(self, train_info, dataset_loader=None, writer=None, step=None, epoch=None, visualize=False, **kwargs):
        # Log training information to tensorboard
        train_info = {k: (v.cpu() if not isinstance(v, float) else v) for k, v in train_info.items()}
        for k, v in train_info.items():
            if not ('loss' in k):
                continue
            assert epoch is not None
            writer.add_scalar('train/' + k, v, epoch)

        if visualize:
            with torch.no_grad():
                print("Visualize generation results: %s" % epoch)
                pcs = self.dec(self.gen(self.get_prior(10).cuda()))
                # np.save("smp.npy", pcs.cpu().numpy())
                # writer.add_mesh('tr_vis/mesh', pcs[:4], global_step=epoch)
                imgs = visualize_point_clouds_3d(pcs)
                writer.add_image('tr_vis/img', torch.as_tensor(imgs), epoch)

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'opt_dis': self.opt_dis.state_dict(),
            'opt_gen': self.opt_gen.state_dict(),
            'opt_dec': self.opt_dec.state_dict(),
            'dis': self.dis.state_dict(),
            'gen': self.gen.state_dict(),
            'dec': self.dec.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)

    def resume(self, path, strict=True, **args):
        # If pretrained AE, then load it up
        # print(self.cfg.trainer.ae_pretrained)
        # ckpt = torch.load(self.cfg.trainer.ae_pretrained)
        # self.dec.load_state_dict(ckpt['dec'], strict=True)
        # self.opt_dec.load_state_dict(ckpt['opt_dec'])

        # gen_ckpt = torch.load(path)
        # self.gen.load_state_dict(gen_ckpt['gen'], strict=True)
        # self.opt_gen.load_state_dict(gen_ckpt['opt_gen'])
        # return 0
    
        ckpt = torch.load(path)
        # self.encoder.load_state_dict(ckpt['enc'], strict=strict)
        self.dec.load_state_dict(ckpt['dec'], strict=strict)
        # self.opt_enc.load_state_dict(ckpt['opt_enc'])
        # self.opt_dec.load_state_dict(ckpt['opt_dec'])
        start_epoch = ckpt['epoch']

        # if 'gen' in ckpt:
            # self.gen.load_state_dict(ckpt['gen'], strict=strict)
        # if 'dis' in ckpt:
            # self.dis.load_state_dict(ckpt['dis'], strict=strict)
        # if 'opt_gen' in ckpt:
            # self.opt_gen.load_state_dict(ckpt['opt_gen'])
        # if 'opt_dis' in ckpt:
            # self.opt_dis.load_state_dict(ckpt['opt_dis'])
        return 0


    # get and save the same prior for shape generation
    def get_prior(self, bs, dim=128):
        gaussian_scale = getattr(self.cfg.models, "gaussian_scale", 1.)
        return torch.randn(bs, dim) * gaussian_scale

    def validate(self, test_loader, idx, evaluation=False, smp=None, ref=None, *args, **kwargs):
        all_res = {}
        max_gen_vali_shape = int(getattr(self.cfg.trainer, "max_gen_validate_shapes", 100))

        # calculate fpd score
        # fpd = self.validate_fpd(test_loader)
        # all_res = {"FPD": fpd}

        if smp is None and ref is None:
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
                    self.dec.eval()
                    z = torch.randn((max_gen_vali_shape, 128)).cuda()
                    w = self.gen(z=z)
                    pcs = self.dec(w)
                    all_smp.append(pcs)
            smp = torch.cat(all_smp, dim=0)[:ref_num]
            smp = normalize_point_clouds(smp)
            ref = normalize_point_clouds(ref)
            np.save(os.path.join(self.cfg.save_dir, 'val', f'smp_{idx}.npy'), smp.detach().cpu().numpy())
            np.save(os.path.join(self.cfg.save_dir, 'val', f'ref_{idx}.npy'), ref.cpu().numpy())
            # sub_sampled = random.sample(range(ref.size(0)), 200)
            # smp = smp[sub_sampled, ...].contiguous()
            # ref = ref[sub_sampled, ...].contiguous()
            print(f"Sample Shape {smp.shape}")
            print(f"Reference Shape {ref.shape}")

            jsd = jsd_between_point_cloud_sets(smp.cpu().numpy(), ref.cpu().numpy())
        else:
            smp = np.load(smp)
            ref = np.load(ref)
            jsd = jsd_between_point_cloud_sets(smp, ref)
            smp = torch.from_numpy(smp).cuda()
            ref = torch.from_numpy(ref).cuda()
        all_res.update({"JSD": jsd})
        print(f"jsd: {jsd}")

        # pdb.set_trace()
        if evaluation:
            # ref_ids = torch.randint(ref_num, (100,))
            # smp_ids = torch.randint(ref_num, (150,))
            # smp_ids = torch.randint(ref_num, (100,))
            # smp = smp[smp_ids]
            # ref = ref[ref_ids]
            gen_res = compute_all_metrics(smp, ref, batch_size=int(getattr(self.cfg.trainer, "val_metrics_batch_size", 100)), accelerated_cd=True)
            all_res.update({("val/gen/%s" % k): (v if isinstance(v, float) else v.item()) for k, v in gen_res.items()})
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

            # for data in tqdm.tqdm(dataloader, desc="Data"):
            #     inp_pts = data['tr_points'].cuda()
            #     all_ref.append(inp_pts)

            smp = torch.cat(all_smp, dim=0)
            # print(smp.max())
            # ref = torch.cat(all_ref, dim=0)
            # import pdb; pdb.set_trace()
            smp = (smp) / (smp.max(dim=1, keepdim=True)[0] - smp.min(dim=1, keepdim=True)[0]) / 2
            # ref = (ref) / (ref.max(dim=1, keepdim=True)[0] - ref.min(dim=1, keepdim=True)[0]) / 2
            # smp = pc_normalize(smp) / 2
            # ref = pc_normalize(ref) / 2
            # smp = (smp - smp.min(dim=1, keepdim=True)[0]) / (smp.max(dim=1, keepdim=True)[0] - smp.min(dim=1, keepdim=True)[0]) / 2
            # ref = (ref - ref.min(dim=1, keepdim=True)[0]) / (ref.max(dim=1, keepdim=True)[0] - ref.min(dim=1, keepdim=True)[0]) / 2
            # smp = (smp) / (smp.max() - smp.min()) / 2
            # ref = (ref) / (ref.max() - ref.min()) / 2
            # smp = (smp - smp.min()) / (smp.max() - smp.min()) / 4
            # ref = (ref - ref.min()) / (ref.max() - ref.min()) / 4
            fpd = calculate_fpd(smp, statistic_save_path = './Frechet/pre_statistics_plane.npz', batch_size=100, dims=1808)
            print(f'Frechet Pointcloud Distance {fpd:.3f}')
            return fpd
