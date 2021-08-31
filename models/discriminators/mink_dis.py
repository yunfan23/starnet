from pdb import set_trace

import MinkowskiEngine as ME
import torch
import torch.nn as nn
import torch.nn.functional as F


class DisMink(ME.MinkowskiNetwork):
    
    # def __init__(self, in_feat, out_feat, D):
    def __init__(self, cfg, cfgmodel):
        super(DisMink, self).__init__(cfgmodel.D)
        self.cfg = cfg
        self.cfgmodel = cfgmodel
        self.in_feat = cfgmodel.in_feat
        self.out_feat = cfgmodel.out_feat
        self.D = cfgmodel.D
        self.use_sigmoid = getattr(cfgmodel, "use_sigmoid", False)

        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=self.in_feat,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=self.D), ME.MinkowskiBatchNorm(64), ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                dimension=self.D), ME.MinkowskiBatchNorm(32), ME.MinkowskiReLU(),
            ME.MinkowskiGlobalPooling(),
            ME.MinkowskiLinear(32, self.out_feat)
        )

        # self.out = ME.MinkowskiLinear(self.out_feat, 1)

    def forward(self, x, return_all=False):
        # coords, _ = ME.utils.sparse_quantize(x, return_index=True)
        x = x * 1000
        # shuffle index
        # set_trace()
        inds = torch.randperm(x.size(1))
        # get first 100 data
        x_ = [x[i][inds[0:100]] for i in range(len(x))]
        coords = ME.utils.batched_coordinates(x_)
        in_feat = torch.ones((len(coords), self.in_feat))
        sin = ME.SparseTensor(
            features=in_feat,
            coordinates=coords,
            device=x.device,
        )

        y = self.net(sin).F
        # set_trace()
        # y = self.out(y).F
        if self.use_sigmoid:
            y = torch.sigmoid(y)

        if return_all:
            return {
                'x': y
            }
        else:
            return y


class PointNet(nn.Module):
    def __init__(self, in_channel, out_channel, embedding_channel=1024):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, embedding_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(embedding_channel)
        self.linear1 = nn.Linear(embedding_channel, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, out_channel, bias=True)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


if __name__ == '__main__':
    model = DisMink()
