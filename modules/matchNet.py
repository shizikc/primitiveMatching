import torch
import torch.nn as nn

from modules.cuboid import CuboidSurface
from modules.pointnet import ClassificationPointNet


class MatchNet(nn.Module):
    def __init__(self, bins):
        super(MatchNet, self).__init__()
        self.bins = bins
        self.encoder = ClassificationPointNet(11 * self.bins)
        # sample points uniformly over [-0.5,0.5]


    def forward(self, x):
        x = self.encoder(x)
        z, q, t, p = torch.split_with_sizes(x,
                                            tuple(torch.tensor([3, 4, 3, 1]) * self.bins), axis=1)
        z = z.view(-1, 1, 3)
        # rotate and translate z by q,t


