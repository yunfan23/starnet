import pdb
import random
import sys

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # nofa: #401
from torch import optim

matplotlib.use('Agg')

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * ((x**2).mean(dim=dim, keepdim=True) + eps).rsqrt()


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def kfn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    dis, idx = pairwise_distance.topk(k=k, dim=-1, largest=False)
    return dis, idx

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    dis, idx = pairwise_distance.topk(k=k, dim=-1)
    return dis, idx


def get_edge_features(x, k, pc=None):
    B, dims, N = x.shape

    if pc is not None:
        dist, idx = knn(pc, k + 1)
    else:
        dist, idx = knn(x, k + 1)
    idx = idx[:, :, 1: k + 1]                    # [B, N, k]
    idx = idx.contiguous().view(B, N * k)

    # gather
    neighbors = []
    for b in range(B):
        tmp = torch.index_select(x[b], 1, idx[b])   # [d, N*k] <- [d, N], 0, [N*k]
        tmp = tmp.view(dims, N, k)
        neighbors.append(tmp)

    neighbors = torch.stack(neighbors)            # [B, d, N, k]

    # centralize
    central = x.unsqueeze(3)                      # [B, d, N, 1]
    central = central.repeat(1, 1, 1, k)          # [B, d, N, k]

    edge = torch.cat([central, neighbors - central], dim=1)
    assert edge.shape == (B, 2 * dims, N, k)
    return edge


def get_opt(params, cfgopt):
    if cfgopt.type == 'adam':
        optimizer = optim.Adam(params, lr=float(cfgopt.lr), betas=(cfgopt.beta1, cfgopt.beta2), weight_decay=cfgopt.weight_decay)
    elif cfgopt.type == 'sgd':
        optimizer = torch.optim.SGD(params, lr=float(cfgopt.lr), momentum=cfgopt.momentum)
    else:
        assert 0, "Optimizer type should be either 'adam' or 'sgd'"

    scheduler = None
    scheduler_type = getattr(cfgopt, "scheduler", None)
    if scheduler_type is not None:
        if scheduler_type == 'exponential':
            decay = float(getattr(cfgopt, "step_decay", 0.1))
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)
        elif scheduler_type == 'multistep':
            milestones = list(getattr(cfgopt, "milestones", [1000]))
            decay = float(getattr(cfgopt, "step_decay", 0.1))
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=decay)
        elif scheduler_type == 'step':
            step_size = int(getattr(cfgopt, "step_epoch", 500))
            decay = float(getattr(cfgopt, "step_decay", 0.1))
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay)
        elif scheduler_type == 'linear':
            step_size = int(getattr(cfgopt, "step_epoch", 2000))
            final_ratio = float(getattr(cfgopt, "final_ratio", 0.1))
            start_ratio = float(getattr(cfgopt, "start_ratio", 0.5))
            duration_ratio = float(getattr(cfgopt, "duration_ratio", 0.45))

            def lambda_rule(ep):
                lr_l = 1.0 - min(1, max(0, ep - start_ratio * step_size) / float(duration_ratio * step_size)) * (1 - final_ratio)
                return lr_l

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        elif scheduler_type == 'cosine_anneal_nocycle':
            final_lr_ratio = float(getattr(cfgopt, "final_lr_ratio", 0.01))
            eta_min = float(cfgopt.lr) * final_lr_ratio
            eta_max = float(cfgopt.lr)

            total_epoch = int(getattr(cfgopt, "step_epoch", 2000))
            start_ratio = float(getattr(cfgopt, "start_ratio", 0.2))
            T_max = total_epoch * (1 - start_ratio)

            def lambda_rule(ep):
                curr_ep = max(0., ep - start_ratio * total_epoch)
                lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * curr_ep / T_max))
                lr_l = lr / eta_max
                return lr_l
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        else:
            assert 0, "args.schedulers should be either 'exponential' or 'linear' or 'step' or 'multistep'"
    return optimizer, scheduler


def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ground_truth_field(prior_points, tr_pts, sigma):
    bs, num_pts = tr_pts.size(0), tr_pts.size(1)
    smp_pts = prior_points.size(1)
    prior_points = prior_points.view(bs, smp_pts, 1, -1)
    tr_pts = tr_pts.view(bs, 1, num_pts, -1)
    dist = (prior_points - tr_pts).norm(dim=3, keepdim=True) ** 2.
    a = - dist / sigma ** 2.
    max_a, _ = torch.max(a, dim=2, keepdim=True)
    diff = torch.exp(a - max_a)
    w_i = diff / diff.sum(dim=2, keepdim=True)

    # (bs, #pts-prior, 1, dim)
    trg_pts = (w_i * tr_pts).sum(dim=2, keepdim=True)
    y = - ((prior_points - trg_pts) / sigma ** 2.).view(bs, smp_pts, -1)
    return y


def ground_truth_reconstruct(inp, sigma, step_size, num_points=2048,
                             num_steps=100, decay=1, interval=10, weight=1):
    with torch.no_grad():
        x = get_prior(inp.size(0), inp.size(1), inp.size(-1)).cuda()
        x_list = []
        x_list.append(x.clone())

        for t in range(num_steps):
            z_t = torch.randn_like(x) * weight
            x += np.sqrt(step_size) * z_t
            grad = ground_truth_field(x, inp, sigma)
            x += 0.5 * step_size * grad
            if t % (num_steps // interval) == 0:
                step_size *= decay
                x_list.append(x.clone())
    return x, x_list


def ground_truth_reconstruct_multi(inp, cfg):
    with torch.no_grad():
        assert hasattr(cfg, "inference")
        step_size_ratio = float(getattr(cfg.inference, "step_size_ratio", 1))
        num_steps = int(getattr(cfg.inference, "num_steps", 5))
        num_points = int(getattr(cfg.inference, "num_points", inp.size(1)))
        weight = float(getattr(cfg.inference, "weight", 1))

        x = get_prior(
            inp.size(0), num_points, cfg.models.scorenet.dim).cuda()
        if hasattr(cfg.trainer, "sigmas"):
            sigmas = cfg.trainer.sigmas
        else:
            sigma_begin = float(cfg.trainer.sigma_begin)
            sigma_end = float(cfg.trainer.sigma_end)
            num_classes = int(cfg.trainer.sigma_num)
            sigmas = np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end),
                                        num_classes))
        x_list = []
        x_list.append(x.clone())
        bs, num_pts = x.size(0), x.size(1)

        for sigma in sigmas:
            sigma = torch.ones((1,)) * sigma
            sigma = sigma.cuda()
            step_size = 2 * sigma ** 2 * step_size_ratio
            for t in range(num_steps):
                z_t = torch.randn_like(x) * weight
                x += torch.sqrt(step_size) * z_t
                grad = ground_truth_field(x, inp, sigma)
                x += 0.5 * step_size * grad
            x_list.append(x.clone())
    return x, x_list


def get_prior(batch_size, num_points=128, inp_dim=128):
    # -1 to 1, uniform
    return (torch.rand(batch_size, num_points, inp_dim) * 2 - 1.) * 1.5


# Visualization
def visualize_point_clouds_3d(pcl_lst, title_lst=None):
    pcl_lst = [pcl.cpu().detach().numpy() for pcl in pcl_lst]
    if title_lst is None:
        title_lst = [""] * len(pcl_lst)

    fig = plt.figure(figsize=(2 * len(pcl_lst), 2))
    for idx, (pts, title) in enumerate(zip(pcl_lst, title_lst)):
        ax = fig.add_subplot(1, len(pcl_lst), 1 + idx, projection='3d')
        ax.set_title(title)
        ax.view_init(8,-85)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, zdir='y')
        ax.set_axis_off()
        # ax1.set_xlim(-1, 1)
        # ax1.set_ylim(-1, 1)
        # ax1.set_zlim(-1, 1)
    fig.canvas.draw()

    # remove axis and autoscale to max
    plt.axis('off')
    plt.autoscale(tight=True)
    plt.savefig("sample.png")

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res

def visualize_point_clouds_3d_3view(pcl_lst, title_lst=None):
    pcl_lst = [pcl.cpu().detach().numpy() for pcl in pcl_lst]
    if title_lst is None:
        title_lst = [""] * len(pcl_lst)

    title = ['left top', 'middle bottom', 'right top']
    fig = plt.figure(figsize=(3 * 5, 4 * len(pcl_lst)))
    for idx, (pts, _) in enumerate(zip(pcl_lst, title_lst)):
    # for idx, (pts, title) in enumerate(zip(pcl_lst, title_lst)):
        for view in range(3):
            elev = [8, -10, 8]
            azim = [-85, -90, -95]
            ax = fig.add_subplot(len(pcl_lst), 3, 1 + view + 3 * idx, projection='3d')
            if idx == 0:
                ax.set_title(title[view])
            ax.set_axis_off()
            ax.view_init(elev[view], azim[view])
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, zdir='y')
            # ax.set_xlim(-0.5, 0.5)
            # ax.set_ylim(-0.5, 0.5)
            # ax.set_zlim(-0.5, 0.5)
            ax.autoscale(tight=True)
    fig.canvas.draw()
    plt.subplots_adjust(wspace=0., hspace=0.)
    plt.autoscale(tight=True)

    # remove axis and autoscale to max
    plt.axis('off')
    plt.autoscale(tight=True)

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res

# Visualization
def visualize_point_clouds_3d_scan(pcl_lst, title_lst=None):
    # pts, gtr, inp):
    pcl_lst = [pcl.cpu().detach().numpy() for pcl in pcl_lst]
    if title_lst is None:
        title_lst = [""] * len(pcl_lst)

    fig = plt.figure(figsize=(3 * len(pcl_lst), 3))
    for idx, (pts, title) in enumerate(zip(pcl_lst, title_lst)):
        ax1 = fig.add_subplot(1, len(pcl_lst), 1 + idx, projection='3d')
        ax1.set_title(title)
        ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 1], s=0.5)
        # print(min(pts[:, 0]), max(pts[:, 0]), min(pts[:, 1]), max(pts[:, 1]), min(pts[:, 2]), max(pts[:, 2]))
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res


# Visualization moving field and likelihood
def get_grid(x, k=10):
    # TODO: set the range of field
    # x = self.get_prior(
    #        1, num_points, self.cfg.models.scorenet.dim).cuda()
    # ind_x = np.arange(x[:,:,0].min(),x[:,:,0].max(),3/k)
    # ind_y = np.arange(x[:,:,1].min(),x[:,:,0].max(),3/k)
    ind_x = np.arange(-1.5, 1.5, 3 / k)
    ind_y = np.arange(-1.5, 1.5, 3 / k)
    X, Y = np.meshgrid(ind_x, ind_y)
    X = torch.tensor(X).view(k * k).to(x)
    Y = torch.tensor(Y).view(k * k).to(x)

    point_grid = torch.ones((1, k * k, 2), dtype=torch.double).to(x)
    point_grid[0, :, 1] = point_grid[0, :, 1] * X
    point_grid[0, :, 0] = point_grid[0, :, 0] * Y
    point_grid = point_grid.float()
    point_grid = point_grid.expand(x.size(0), -1, -1)
    return point_grid


def visualize_point_clouds_2d_overlay(pcl_lst, title_lst=None, path=None):
    # pts, gtr, inp):
    pcl_lst = [pcl.cpu().detach().numpy() for pcl in pcl_lst]
    if title_lst is None:
        title_lst = [""] * len(pcl_lst)

    fig = plt.figure(figsize=(3, 3))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title(title_lst[0])
    for idx, pts in enumerate(pcl_lst):
        ax1.scatter(pts[:, 0], pts[:, 1], s=5)
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))
    if path:
        plt.savefig(path)
    plt.close()
    return res


def visualize_field(gtr, grid, field, k, label='field'):
    grid_ = np.reshape(grid.cpu().detach().numpy(), (1, k * k, 2))
    if field.size(-1) == 2:
        field = np.reshape(field.cpu().detach().numpy(), (1, k * k, 2))
        fig = plt.figure(figsize=(int(k / 100) * 2, int(k / 100)))
        plt.title(label)
        cs = fig.add_subplot(1, 2, 1)
        field_val = np.sqrt(np.reshape((field ** 2).sum(axis=-1), (1, k * k, 1)))
    else:
        fig = plt.figure(figsize=(int(k / 100), int(k / 100)))
        plt.title(label)
        cs = fig.add_subplot(1, 1, 1)
        field_val = np.reshape(field.cpu().detach().numpy(), (1, k * k, 1))

    gt = gtr.cpu().detach().numpy()

    for i in range(np.shape(field_val)[0]):
        # cs = fig.add_subplot(1, 2, 1)
        X = np.reshape(grid_[i, :, 0], (k, k))
        Y = np.reshape(grid_[i, :, 1], (k, k))
        cs.contourf(X, Y, np.reshape(field_val[i, :], (k, k)), 20,
                    vmin=min(field_val[i, :]), vmax=max(field_val[i, :]),
                    cmap=cm.coolwarm)
        print(min(field_val[i, :]), max(field_val[i, :]))
        m = plt.cm.ScalarMappable(cmap=cm.coolwarm)
        m.set_array(np.reshape(field_val[i, :], (k, k)))
        m.set_clim(min(field_val[i, :]), max(field_val[i, :]))

    if np.shape(field)[-1] == 2:
        for i in range(np.shape(field_val)[0]):
            ax = fig.add_subplot(1, 2, 2)
            scale = 20
            indx = np.array([np.arange(0, k, scale) + t * k for t in range(0, k, scale)])
            X = np.reshape(grid_[i, indx, 0], int(k * k / scale / scale))
            Y = np.reshape(grid_[i, indx, 1], int(k * k / scale / scale))
            u = np.reshape(field[i, indx, 0], int(k * k / scale / scale))
            v = np.reshape(field[i, indx, 1], int(k * k / scale / scale))

            color = np.sqrt(v ** 2 + u ** 2)
            field_norm = field / field_val
            u = np.reshape(field_norm[i, indx, 0], int(k * k / scale / scale))
            v = np.reshape(field_norm[i, indx, 1], int(k * k / scale / scale))
            ax.quiver(X, Y, u, v, color, alpha=0.8, cmap=cm.coolwarm)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.set_aspect('equal')
            ax.scatter(gt[:, 0], gt[:, 1], s=1, color='r')

    fig.canvas.draw()
    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res


def visualize_procedure(sigmas, fig_list, gtr, num_vis, cfg, name="Rec_gt"):
    all_imgs = []
    sigmas = np.append([0], sigmas)
    for idx in range(num_vis):
        img = visualize_point_clouds_3d(
            [fig_list[i][idx] for i in
             range(0, len(fig_list), 1)] + [gtr[idx]],
            [(name + " step" +
              str(i * int(getattr(cfg.inference, "num_steps", 5))) +
              " sigma%.3f" % sigmas[i])
             for i in range(0, len(fig_list), 1)] + ["gt shape"])
        all_imgs.append(img)
    img = np.concatenate(all_imgs, axis=1)
    return img


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
    	# Update the i-th farthest point
        centroids[:, i] = farthest
        # Take the xyz coordinate of the farthest point
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # Calculate the Euclidean distance from all points in the point set to this farthest point
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # Update distances to record the minimum distance of each point in the sample from all existing sample points
        mask = dist < distance
        distance[mask] = dist[mask]
        # Find the farthest point from the updated distances matrix, and use it as the farthest point for the next iteration
        farthest = torch.max(distance, -1)[1]
    return centroids


def normalize_point_clouds(pcs, mode="shape_bbox"):
    import tqdm

    # for i in tqdm.tqdm(range(pcs.size(0)), desc='Normalize'):
    for i in range(pcs.size(0)):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        else:
            shift = 0.
            scale = 1.
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs


def pc_normalize(pc):
    """ pc: BxNxC, return BxNxC """
    centroid = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=2, keepdim=True)), dim=1)[0]
    m = m.unsqueeze(1).expand_as(pc)
    pc = pc / m / 2
    return pc


def to_precision(x, p=4):
    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)


def count_parameters(network):
    return sum(p.numel() for p in network.parameters())
