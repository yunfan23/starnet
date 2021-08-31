import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.num_pts = cfg.num_pts
        self.fc1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(256, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool1d(self.num_pts, 1)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batchsize = x.size()[0]
        x = x.transpose(1, 2)
        x1 = self.fc1(x)
        x2 = self.maxpool(x1)
        x2 = x2.view(batchsize, 1024)
        x3 = self.mlp(x2)

        return x3
