import torch.nn as nn
import torch.nn.functional as F
from trainers.utils.utils import normalize_2nd_moment


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.z_dim = cfg.z_dim
        self.w_dim = cfg.w_dim
        self.hidden = 256

        self.fc1 = nn.Linear(self.z_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.hidden)
        self.fc4 = nn.Linear(self.hidden, self.w_dim)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
            
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)

        return x
