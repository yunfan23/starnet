import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        cfg,
        num_points: int = 2048,
        bottleneck_size: int = 128
    ):
        super(Decoder, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.decoder = AdaInPointGenCon(input_dim=2, style_dim=self.bottleneck_size)

    def forward(self, style):
        outs = []
        regular_grid = torch.cuda.FloatTensor(grid_generation(self.num_points))
        regular_grid = regular_grid.transpose(0, 1).contiguous().unsqueeze(0)
        regular_grid = regular_grid.expand(style.size(0), regular_grid.size(1), regular_grid.size(2)).contiguous()
        regular_grid = ((regular_grid - 0.5) * 2).contiguous()

        outs = self.decoder(regular_grid, style)
        outs = outs.transpose(1, 2).contiguous()
        return outs

class AdaInPointGenCon(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        style_dim: int = 1024,
        hidden_size: int = 1024
    ):
        super(AdaInPointGenCon, self).__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.style_dim = style_dim
        self.dec = GridDecoder(self.input_dim, self.hidden_size)

        # MLP to generate AdaIN parameters
        self.mlp = nn.Sequential(
            nn.Linear(self.style_dim, get_num_adain_params(self.dec)),
            nn.ReLU(),
        )

    def forward(self, content, style):
        adain_params = self.mlp(style)
        assign_adain_params(adain_params, self.dec)
        return self.dec(content)

class SELayer1D(nn.Module):
    """
    input:
        x:(b, c, m)

    output:
        out:(b, c, m')
    """

    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        (b, c, _) = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


def grid_generation(num_points):
    num_points = num_points
    grain_x = 2 ** np.floor(np.log2(num_points) / 2) - 1
    grain_y = 2 ** np.ceil(np.log2(num_points) / 2) - 1

    vertices = []
    for i in range(int(grain_x + 1)):
        for j in range(int(grain_y + 1)):
            vertices.append([i / grain_x, j / grain_y])

    return vertices


def get_num_adain_params(model):
    """
    input:
    - model: nn.module

    output:
    - num_adain_params: int
    """
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            num_adain_params += 2 * m.num_features
    return num_adain_params


def assign_adain_params(adain_params, model):

    """
    inputs:
    - adain_params: b x parameter_size
    - model: nn.module

    function:
    assign_adain_params
    """
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            mean = adain_params[:, : m.num_features]
            std = adain_params[:, m.num_features : 2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[:, 2 * m.num_features :]


class AdaptiveInstanceNorm1d(nn.Module):
    """
    input:
    - inp: (b, c, m)

    output:
    - out: (b, c, m')
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class GridDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_size: int = 1024
    ):
        super(GridDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.conv1 = torch.nn.Conv1d(self.input_dim, self.hidden_size, 1)
        self.conv2 = torch.nn.Conv1d(self.hidden_size, self.hidden_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.hidden_size // 2, self.hidden_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.hidden_size // 4, 3, 1)
        self.th = nn.Tanh()

        self.adain1 = AdaptiveInstanceNorm1d(self.hidden_size)
        self.adain2 = AdaptiveInstanceNorm1d(self.hidden_size // 2)
        self.adain3 = AdaptiveInstanceNorm1d(self.hidden_size // 4)

        self.bn1 = torch.nn.BatchNorm1d(self.hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.hidden_size // 4)


    def forward(self, x):
        x = F.relu(self.bn1(self.adain1(self.conv1(x))))
        x = F.relu(self.bn2(self.adain2(self.conv2(x))))
        x = F.relu(self.bn3(self.adain3(self.conv3(x))))
        x = self.th(self.conv4(x))
        return x
