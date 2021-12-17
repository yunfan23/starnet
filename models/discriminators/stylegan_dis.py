import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, cfg):
        self.layer_num = len(cfg.features) - 1
        super(Discriminator, self).__init__()

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(cfg.features[inx], cfg.features[inx + 1], kernel_size=1, stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.final_layer = nn.Sequential(nn.Linear(2 * cfg.features[-1], cfg.features[-1]),
                                         nn.Linear(cfg.features[-1], cfg.features[-2]),
                                         nn.Linear(cfg.features[-2], cfg.features[-2]),
                                         nn.Linear(cfg.features[-2], 1))

    def forward(self, f):
        feat = f.transpose(1, 2)
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)

        out_avg = F.avg_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        out_max = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        out = torch.cat((out_avg, out_max), dim=-1)
        out = self.final_layer(out)

        return out
