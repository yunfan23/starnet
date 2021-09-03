import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decoders.sparenet_decoder import AdaptiveInstanceNorm1d, SELayer1D


class Decoder(nn.Module):
    def __init__(
        self,
        cfg,
        input_dim: int = 128,
        hidden_size: int = 1024
    ):
        super(Decoder, self).__init__()
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

        self.se1 = SELayer1D(channel=self.hidden_size)
        self.se2 = SELayer1D(channel=self.hidden_size // 2)
        self.se3 = SELayer1D(channel=self.hidden_size // 4)

    def forward(self, x):
        import pdb; pdb.set_trace()
        x = F.relu(self.se1(self.bn1(self.adain1(self.conv1(x)))))
        x = F.relu(self.se2(self.bn2(self.adain2(self.conv2(x)))))
        x = F.relu(self.se3(self.bn3(self.adain3(self.conv3(x)))))
        x = self.th(self.conv4(x))
        return x
