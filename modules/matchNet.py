import torch
import torch.nn as nn

from modules.cuboid import sample_cudoid, rotate_cuboid, get_cuboid_corner
from modules.pointnet import PointNetCls
from modules.pointnet_pp import pointNetPP


class MatchNet(nn.Module):
    def __init__(self, bins, samplesPerFace, dev):
        super(MatchNet, self).__init__()
        self.bins = bins  # per face partition
        self.encoderBlock1 = PointNetCls(self.bins ** 3)
        # self.cnn3dBlock = CNN3D(self.bins ** 9)
        # self.encoderBlock1 = pointNetPP(self.bins ** 3)

        self.samplesPerFace = samplesPerFace
        self.dev = dev
        self.samples = sample_cudoid(1, self.bins ** 3, self.samplesPerFace).to(self.dev)

    def forward(self, x, add_samples=False):
        """

        :param sample:
        :param x: in torch.Size(bs, nPoints, 3)
        :return: pred in (bs, nCuboid, 3*nSamplePerFace, 3)
        """
        # bs = x.shape[0]

        # uniformly sampled cuboids in [-1,1], shape torch.Size([bs, bins**3, nSamples, 3])
        # samples = self.samples.repeat(bs, 1, 1, 1)

        x, p = self.encoderBlock1(x)  # bs x bins**3, bs x bins**3 in [0,1]


        # z, q, t, p = torch.split_with_sizes(x,
        #                                     tuple(torch.tensor([3, 4, 3, 1]) * (self.bins ** 3)), axis=1)
        #
        # z = z.reshape(bs, -1, 3).unsqueeze(2).repeat(1, 1, 3 * self.samplesPerFace,
        #                                              1)  # bs x bins^3 x nSaples x 3
        # t = t.reshape(bs, -1, 3).unsqueeze(2).repeat(1, 1, 3 * self.samplesPerFace,
        #                                              1)  # bs x bins^3 x nSaples x 3
        # q = q.reshape(bs, -1, 4)
        #
        # if add_samples:
        #     # rotate by q
        #     q = F.normalize(q, p=2, dim=2)
        #     q = rotate_cuboid(q).unsqueeze(2)  # (bs, nCubiods, 3, 3)
        #     out = samples.unsqueeze(4)
        #     out = torch.matmul(q, out).squeeze(4)
        #
        #     # move to voxels
        #     b = get_cuboid_corner(self.bins).to(self.dev)  # bins**3 x 3
        #     b = b.unsqueeze(0).repeat(bs, 1, 1)  # bs x bins**3 x 3
        #     out = out * (1 / self.bins) + b.unsqueeze(2)
        #
        #     # translate by t, scale by z
        #     out = out * z + t
        #     return out, p, z
        return None, p, None


if __name__ == '__main__':
    batch_size = 2
    bins_per_face = 2
    ncuboid = bins_per_face ** 3
    samples_per_face = 15
    sim_data = torch.rand(batch_size, 1792, 3)

    mn = MatchNet(bins_per_face, samples_per_face, "cpu")
    o, p, z = mn(sim_data)

    import numpy.random as rm

    # from utils.visualization import plot_pc_mayavi

    c = [tuple(rm.random(3)) for i in range(ncuboid)]
    # plot_pc_mayavi([mn.samples[0][i] for i in range(ncuboid)], colors=c)

    # plot_pc_mayavi([o[0][i].detach().numpy() for i in range(ncuboid)], colors=c)
