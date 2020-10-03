import torch
import torch.nn as nn

from modules.cuboid import sample_partial_cuboid
from modules.pointnet import ClassificationPointNet


class MatchNet(nn.Module):
    def __init__(self, bins, samplesPerFace):
        super(MatchNet, self).__init__()
        self.bins = bins  # per face partition
        self.encoder = ClassificationPointNet(11 * (self.bins ** 3))
        self.samplesPerFace = samplesPerFace
        # uniformly sampled cuboids in [-1,1], shape torch.Size([1, bins**3, nSamples, 3])
        self.samples = sample_partial_cuboid(1, self.bins, self.samplesPerFace)

    def forward(self, x):
        """

        :param x: in torch.Size(bs, nPoints, 3)
        :return:
        """
        bs = x.shape[0]
        # uniformly sampled cuboids in [-1,1], shape torch.Size([bs, bins**3, nSamples, 3])
        samples = self.samples.repeat(bs, 1, 1, 1)
        x, _ = self.encoder(x)  # bs x 11*bins**3
        z, q, t, p = torch.split_with_sizes(x,
                                            tuple(torch.tensor([3, 4, 3, 1]) * (self.bins ** 3)), axis=1)

        z = z.reshape(z.shape[0], -1, 3).unsqueeze(2).repeat(1, 1, 3*self.samplesPerFace, 1) * 0.5  # bs x bins^3 x nSaples x 3
        t = t.reshape(t.shape[0], -1, 3).unsqueeze(2).repeat(1, 1, 3*self.samplesPerFace, 1) * 0.5  # bs x bins^3 x nSaples x 3
        # rotate and translate samples by z by t
        out = samples * z + t
        # TODO : rotate out by q (bs x 4)

        return out, p, z


if __name__ == '__main__':
    batch_size = 2
    bins_per_face = 5
    ncuboid = bins_per_face ** 3
    samples_per_face = 333
    sim_data = torch.rand(batch_size, 1792, 3)

    mn = MatchNet(bins_per_face, samples_per_face)
    out, p, z = mn(sim_data)

    import numpy.random as rm
    from utils.visualization import plot_pc_mayavi

    c = [tuple(rm.random(3)) for i in range(ncuboid)]
    plot_pc_mayavi([mn.samples[0][i] for i in range(ncuboid)], colors=c)

    plot_pc_mayavi([out.detach().numpy()[0][i] for i in range(ncuboid)], colors=c)
