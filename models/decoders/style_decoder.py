import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from trainers.utils.utils import (get_edge_features, knn, next_power_of_2,
                                  normalize_2nd_moment)


class Generator(nn.Module):
    # def __init__(self, num_pts, num_k, z_dim, w_dim, use_se=True, use_noise=True):
    def __init__(self, cfg):
        super().__init__()
        self.num_pts = cfg.num_pts
        self.num_k = cfg.num_k
        self.z_dim = cfg.z_dim
        self.w_dim = cfg.w_dim
        self.use_noise = cfg.use_noise
        self.fuseconv = cfg.fuseconv
        self.use_1d = cfg.use_1d
        self.synthesis = SynthesisNetwork(
            self.w_dim,
            self.num_pts,
            self.num_k,
            use_1d=self.use_1d,
            fuseconv = self.fuseconv,
            use_noise=self.use_noise
        )
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(self.z_dim, self.w_dim, self.num_ws)

    def forward(self, z, c=None, **synthesis_kwargs):
        ws = self.mapping(z)
        pc = self.synthesis(ws, **synthesis_kwargs)
        return pc


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, num_ws):
        super().__init__()
        self.w_dim      = w_dim
        self.z_dim      = z_dim
        self.layers     = []
        self.bns        = []
        self.num_layers = 6
        self.num_ws     = num_ws
        self.hidden     = z_dim

        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(self.hidden, self.hidden))
            self.bns.append(nn.BatchNorm1d(self.hidden))

        self.layers = nn.ModuleList(self.layers)
        self.bns = nn.ModuleList(self.bns)
        self.out = nn.Linear(self.hidden, self.w_dim)
        self.out_bn = nn.BatchNorm1d(self.z_dim)

    def forward(self, z=None, c=None):
        x = normalize_2nd_moment(z.to(torch.float32))
        for layer, bn in zip(self.layers, self.bns):
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
        x = self.out(x)
        x = self.out_bn(x)

        # Broadcast
        if self.num_ws is not None:
            x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        return x


class SynthesisNetwork(nn.Module):
    def __init__(
        self,
        w_dim,
        num_pts=512,
        num_k=20,
        use_noise=False,
        use_1d=False,
        fuseconv=False,
        channel_base=16384,    # Overall multiplier for the number of channels.
        channel_max=256        # Maximum number of channels in any layer.
    ):

        super(SynthesisNetwork, self).__init__()
        self.w_dim = w_dim
        self.num_k = num_k
        self.fuseconv = fuseconv
        self.init_pts = 128
        self.num_pts = next_power_of_2(num_pts)
        self.num_blks = int(math.log2(self.num_pts / self.init_pts)) + 1
        self.blk_res = [self.init_pts * 2 ** i for i in range(self.num_blks)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.blk_res}

        self.num_ws = 0
        for res in self.blk_res:
            in_channels = channels_dict[res // 2] if res > self.init_pts else 0
            out_channels = channels_dict[res]
            is_last = (res == self.num_pts)
            block = SynthesisBlock(
                in_channels,
                out_channels,
                w_dim,
                num_k=self.num_k,
                num_pts=res,
                is_last=is_last,
                use_noise=use_noise,
                use_1d=use_1d
            )
            # increase neighbor
            # self.num_k *= 2
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_topc
            setattr(self, f'b{res}', block)

    def forward(self, w):
        w_idx = 0
        blk_w = []
        x = pc = None

        for res in self.blk_res:
            blk = getattr(self, f'b{res}')
            blk_w.append(w.narrow(1, w_idx, blk.num_conv + blk.num_topc))
            w_idx += blk.num_conv

        for res, cur_w in zip(self.blk_res, blk_w):
            blk = getattr(self, f'b{res}')
            x, pc = blk(x, pc, cur_w)

        return pc


class SynthesisBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        num_pts,
        is_last,
        num_k=20,
        use_noise=False,
        up=2,
        mode="nearest",
        clamp=1,
        fuseconv=False,
        use_1d=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_pts = num_pts
        self.num_k = num_k
        self.use_noise = use_noise
        self.use_1d = use_1d
        self.w_dim = w_dim
        self.num_conv = 0
        self.num_topc = 0
        self.architecture = 'skip'
        self.is_last = is_last
        self.up = 2
        self.mode = mode
        self.clamp = clamp

        if self.in_channels == 0:
            self.conv1 = ConstantInput(out_channels, num_pts)
            self.num_conv += 1
        else:
            if fuseconv:
                self.conv1 = UpsampleEdgeconv(in_channels, out_channels, num_k)
            else:
                self.up = Upsample(num_k, up, mode)
                if use_1d:
                    self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
                else:
                    self.conv1 = EdgeConv(in_channels, out_channels, num_k)
            self.num_conv += 1

        self.noise1 = NoiseLayer(out_channels, use_noise)
        self.adain1 = AdaptiveInstanceNorm(out_channels, w_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        if not use_1d:
            self.conv2 = EdgeConv(out_channels, out_channels, num_k)
        else:
            self.conv2 = nn.Conv1d(out_channels, out_channels, 1)

        self.num_conv += 1
        self.noise2 = NoiseLayer(out_channels, use_noise)
        self.adain2 = AdaptiveInstanceNorm(out_channels, w_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

        if is_last or self.architecture == 'skip':
            self.pc_up = Upsample(num_k)
            self.to_pc = ToPCLayer(out_channels, w_dim)
            self.num_topc += 1

    def forward(self, content, pc, style):
        if content is None:
            content = style
        x = content
        if not self.in_channels == 0:
            x = self.up(x, pc)
            pc = self.pc_up(pc)

        if not self.use_1d:
            # x = self.conv1(x, pc)
            x = self.conv1(x, None)
        else:
            x = self.conv1(x)
        x = self.adain1(self.lrelu1(self.noise1(x)), style[:, 0])

        if not self.use_1d:
            # x = self.conv2(x, pc)
            x = self.conv2(x, None)
        else:
            x = self.conv2(x)
        x = self.adain2(self.lrelu2(self.noise2(x)), style[:, 1])
        
        if self.is_last or self.architecture == 'skip':
            y = self.to_pc(x, style[:, -1])
            pc = pc.add_(y) if pc is not None else y

        return x, pc


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_k):
        super(EdgeConv, self).__init__()
        self.num_k = num_k

        self.conv = nn.Conv2d(2 * in_channels, out_channels, [1, num_k], 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, pc=None):
        x = get_edge_features(x, self.num_k, pc)
        x = self.lrelu(self.bn(self.conv(x)))
        x = x.squeeze(3)
        return x


class ToPCLayer(nn.Module):
    def __init__(self, in_channels, w_dim, pc_dim=3, clamp=1):
        super(ToPCLayer, self).__init__()
        self.pc_dim = pc_dim
        self.clamp = clamp
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, 128, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 64, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, pc_dim, 1),
            nn.LeakyReLU(0.2)
        )
        self.norm = nn.InstanceNorm1d(in_channels)
        self.style = nn.Linear(w_dim, in_channels * 2)
        self.style.bias.data[: in_channels] = 1
        self.style.bias.data[in_channels:] = 0

    def forward(self, content, style):
        style = self.style(style).unsqueeze(2)
        gamma, beta = style.chunk(2, 1)
        x = self.norm(content)
        x = gamma * x + beta
        x = self.fc(x)
        return x


class ConstantInput(nn.Module):
    def __init__(self, channels, num_pts=4, random_input=True):
        super().__init__()
        
        pts = torch.arange(0., 1., 1.0 / num_pts)
        vertices = pts.expand(1, channels, num_pts)
        if random_input:
            self.const = nn.Parameter(torch.randn(1, channels, num_pts))
        else:
            self.const = nn.Parameter(vertices)

    def forward(self, content, pc=None):
        batch = content.shape[0]
        out = self.const.repeat(batch, 1, 1)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if channel // reduction == 0:
            channel_red = 1
        else:
            channel_red = channel // reduction
        self.fc = nn.Sequential(
            nn.Linear(channel, channel_red, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_red, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channels, w_dim):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim

        # change to 1d normalization
        self.norm = nn.InstanceNorm1d(self.in_channels)
        self.style = nn.Linear(self.w_dim, self.in_channels * 2)

        self.style.bias.data[:self.in_channels] = 1
        self.style.bias.data[self.in_channels:] = 0

    def forward(self, content, style):
        style = self.style(style).unsqueeze(2)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(content)
        out = gamma * out + beta

        return out


class NoiseLayer(nn.Module):
    def __init__(self, channels, use_noise):
        super().__init__()
        # weight is the strengh of the noise applied
        self.weight = nn.Parameter(torch.zeros(channels))
        self.use_noise = use_noise
        self.noise = None

    def forward(self, x, noise=None):
        if not self.use_noise:
            return x
        if noise is None and self.noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype)
        elif noise is None:
            noise = self.noise
        x = x + self.weight.view(1, -1, 1) * noise
        return x


class Upsample(nn.Module):
    def __init__(self, num_k=2, scale_factor=2, mode="knn"):
        super().__init__()
        self.num_k = num_k

    def forward(self, x, pc=None):
        B, dims, N = x.shape
        device = x.device
       
        if pc is not None:
            dist, idx = knn(pc, self.num_k + 1)
        else:
            dist, idx = knn(x, self.num_k + 1)
        idx = idx[:, :, 1:]                                 # [B, N, k]
        dist = dist[:, :, 1:]                               # [B, N, k]

        idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
        idx = idx + idx_base
        idx = idx.view(-1)

        x = x.transpose(2, 1).contiguous()                  # [B, N, d]
        feature = x.view(B * N, -1)[idx, :]
        feature = feature.view(B, N, self.num_k, dims)      # [B, N, k, d]
        feature = feature.permute(0, 1, 3, 2)               # [B, N, d, k]

        # centralize
        central = x.unsqueeze(3)                            # [B, N, d, 1]
        central = central.repeat(1, 1, 1, self.num_k)       # [B, N, d, k]
        edge = feature - central                            # [B, N, d, k]

        # get the weight based on distance
        # weight = F.softmax(dist, dim=-1).unsqueeze(3)       # [B, N, k, 1]
        # neighbor = torch.matmul(edge, weight).squeeze(3)    # [B, N, d]
        
        # take average to form a new neighbor
        neighbor = edge.mean(dim=3)
        assert neighbor.shape == (B, N, dims)

        # append new x with original x
        x_up = torch.cat([x, neighbor + x], dim=1)
        x_up = x_up.transpose(1, 2)
        assert x_up.shape == (B, dims, 2 * N)
        return x_up


class UpsampleEdgeconv(nn.Module):
    def __init__(self, Fin, Fout, k, num):
        super(UpsampleEdgeconv, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.num = num
        
        self.conv2 = conv2dbr(2*Fin, 2*Fout, [1, 2 * k], [1, 1])
        self.inte_conv_hk = nn.Sequential(
            nn.Conv2d(2 * Fin, 4 * Fin, [1, k // 2 + 1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(4 * Fin),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, pc=None):
        B, Fin, N = x.shape
        x = get_edge_features(x, self.num_k, pc)                        # [B, 2Fin, N, k]

        BB, CC, NN, KK = x.size()
        inte_x = self.inte_conv_hk(x)                                   # Bx2CxNxk/2
        inte_x = inte_x.transpose(2, 1)                                 # BxNx2Cxk/2
        inte_x = inte_x.contiguous().view(BB, NN, CC, 2, KK//2)         # BxNxCx2x(k//2+1)
        inte_x = inte_x.contiguous().view(BB, NN, CC, KK)               # BxNxCx(k+2)
        inte_x = inte_x.permute(0, 2, 1, 3)                             # BxCxNxk
        merge_x = torch.cat((x, inte_x), 3)                             # BxCxNx2k

        x = self.conv2(merge_x)                                         # [B, 2*Fout, N, 1]
        x = x.unsqueeze(3)                                              # BxkcxN
        x = x.contiguous().view(B, self.Fout, 2, N)
        x = x.contiguous().view(B, self.Fout, 2 * N)

        assert x.shape == (B, self.Fout, 2 * N)
        return x


class conv2dbr(nn.Module):
    """ Conv2d-bn-relu
    [B, Fin, H, W] -> [B, Fout, H, W]
    """
    def __init__(self, Fin, Fout, kernel_size, stride=1):
        super(conv2dbr, self).__init__()
        self.conv = nn.Conv2d(Fin, Fout, kernel_size, stride)
        self.bn = nn.BatchNorm2d(Fout)
        self.ac = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x) # [B, Fout, H, W]
        x = self.bn(x)
        x = self.ac(x)
        return x
