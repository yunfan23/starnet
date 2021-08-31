import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.z_dim = cfg.z_dim
        self.w_dim = cfg.w_dim
        self.hidden = 512

        self.fc1 = nn.Linear(self.z_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        self.fc_m = nn.Linear(self.hidden, self.w_dim)
        self.fc_v = nn.Linear(self.hidden, self.w_dim)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        m = self.fc_m(x)
        v = self.fc_v(x)

        return m, v
