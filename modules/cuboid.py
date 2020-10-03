import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.visualization import plot_pc_mayavi


class CuboidSurface(nn.Module):
    def __init__(self, nSamples):
        self.nSamples = (nSamples // 3) * 3
        self.samplesPerFace = nSamples // 3
        self.faces = None

    def sample_points_cuboid_uniform(self, primShapes):
        """

        :param primShapes: tensor shape bs x 1 x 3 where dim 2 is z_m = (w_m, h_m, d_m)
        :return: output B x nSamples x 3 in [-1, 1]
        """
        #
        bs = primShapes.size(0)
        nsp = self.samplesPerFace

        data_type = primShapes.data.type()
        coeffBernoulli = torch.bernoulli(torch.Tensor(bs, nsp, 3).fill_(0.5)).type(data_type)  # makes entries 0 and 1
        coeffBernoulli = 2 * coeffBernoulli - 1  # makes entries -1 and 1

        coeff_w = torch.Tensor(bs, nsp, 3).type(data_type).uniform_(-1, 1)
        coeff_w[:, :, 0].copy_(coeffBernoulli[:, :, 0].clone())  # vs, nsp, 1 in {-1, 1}

        coeff_h = torch.Tensor(bs, nsp, 3).type(data_type).uniform_(-1, 1)
        coeff_h[:, :, 1].copy_(coeffBernoulli[:, :, 1].clone())

        coeff_d = torch.Tensor(bs, nsp, 3).type(data_type).uniform_(-1, 1)
        coeff_d[:, :, 2].copy_(coeffBernoulli[:, :, 2].clone())

        coeff = torch.cat([coeff_w, coeff_h, coeff_d], dim=1)
        coeff = Variable(coeff)
        samples = self.sample(primShapes, coeff)
        importance_weights = self.sample_wt_module(primShapes)
        return samples, importance_weights

    def sample_points_cuboid_prop(self, primShapes):
        """

        :param primShapes: tensor shape bs x 1 x 3 where dim 2 is z_m = (w_m, h_m, d_m)
        :return: output B x nSamples x 3
        """
        #
        bs = primShapes.size(0)
        nsp_w, nsp_h, nsp_d = (cs.multi_dist(dims).flatten() * self.nSamples).round()
        nsp_w, nsp_h, nsp_d = int(nsp_w), int(nsp_h), int(nsp_d)

        data_type = primShapes.data.type()

        coeffBernoulli = torch.bernoulli(torch.Tensor(bs, nsp_w, 3).fill_(0.5)).type(data_type)  # makes entries 0 and 1
        coeffBernoulli = 2 * coeffBernoulli - 1  # makes entries -1 and 1

        coeff_w = torch.Tensor(bs, nsp_w, 3).type(data_type).uniform_(-1, 1)
        coeff_w[:, :, 0].copy_(coeffBernoulli[:, :, 0].clone())  # vs, nsp, 1 in {-1, 1}

        coeffBernoulli = torch.bernoulli(torch.Tensor(bs, nsp_h, 3).fill_(0.5)).type(data_type)  # makes entries 0 and 1
        coeffBernoulli = 2 * coeffBernoulli - 1  # makes entries -1 and 1

        coeff_h = torch.Tensor(bs, nsp_h, 3).type(data_type).uniform_(-1, 1)
        coeff_h[:, :, 1].copy_(coeffBernoulli[:, :, 1].clone())

        coeffBernoulli = torch.bernoulli(torch.Tensor(bs, nsp_d, 3).fill_(0.5)).type(data_type)  # makes entries 0 and 1
        coeffBernoulli = 2 * coeffBernoulli - 1  # makes entries -1 and 1

        coeff_d = torch.Tensor(bs, nsp_d, 3).type(data_type).uniform_(-1, 1)
        coeff_d[:, :, 2].copy_(coeffBernoulli[:, :, 2].clone())

        coeff = torch.cat([coeff_w, coeff_h, coeff_d], dim=1)
        coeff = Variable(coeff)
        samples = self.sample(primShapes, coeff)
        importance_weights = self.sample_wt_module(primShapes)
        return samples, importance_weights

    def sample(self, dims, coeff):
        dims_rep = dims.repeat(1, self.nSamples, 1)
        return dims_rep * coeff

    def cuboidAreaModule(self, dims):
        """
        :param: dims:
        :return: tensor in shape (1, self.nSamples, 1) where dim 2 is cuboid area
        """

        width, height, depth = torch.chunk(dims, chunks=3, dim=2)

        wh = width * height
        hd = height * depth
        wd = width * depth

        surfArea = 2 * (wh + hd + wd)
        areaRep = surfArea.repeat(1, self.nSamples, 1)
        return areaRep

    def sample_wt_module(self, dims):
        # dims is bs x 1 x 3
        area = self.cuboidAreaModule(dims)  # bs x 1 x 1
        dimsInv = dims.pow(-1)
        dimsInvNorm = dimsInv.sum(2).repeat(1, 1, 3)
        normWeights = 3 * (dimsInv / dimsInvNorm)

        widthWt, heightWt, depthWt = torch.chunk(normWeights, chunks=3, dim=2)
        widthWt = widthWt.repeat(1, self.samplesPerFace, 1)
        heightWt = heightWt.repeat(1, self.samplesPerFace, 1)
        depthWt = depthWt.repeat(1, self.samplesPerFace, 1)

        sampleWt = torch.cat([widthWt, heightWt, depthWt], dim=1)
        finalWt = (1 / self.samplesPerFace) * (sampleWt * area)
        return finalWt

    # def multi_dist(self, dims):
    #     """
    #
    #     :param dims: Bs x 1 x 3
    #     :return: Bs x 1 x 3
    #     """
    #     width, height, depth = torch.chunk(dims, chunks=3, dim=2)
    #
    #     wh = width * height
    #     hd = height * depth
    #     wd = width * depth
    #
    #     surfArea = (wh + hd + wd)
    #     return torch.tensor([hd, wd, wh]) / surfArea  # .repeat(1, self.nSamples, 1)


def sample_cudoid(bs, nCuboid, nSamplePerFace):
    """

    :param nSamplePerFace:
    :return: samples,shape (bs, nCuboid, 3*nSamplePerFace, 3) in [-1,1]^3
    """
    # fix 2/6 Faces -1 and 1
    coeffBernoulli = torch.bernoulli(torch.Tensor(bs, nCuboid, nSamplePerFace, 3).fill_(0.5))  # random entries 0 and 1
    coeffBernoulli = 2 * coeffBernoulli - 1  # makes entries -1 and 1

    coeff_w = torch.Tensor(bs, nCuboid, nSamplePerFace, 3).uniform_(-1, 1)
    coeff_w[:, :, 0].copy_(coeffBernoulli[:, :, 0].clone())  # vs, nsp, 1 in {-1, 1}

    coeff_h = torch.Tensor(bs, nCuboid, nSamplePerFace, 3).uniform_(-1, 1)
    coeff_h[:, :, 1].copy_(coeffBernoulli[:, :, 1].clone())

    coeff_d = torch.Tensor(bs, nCuboid, nSamplePerFace, 3).uniform_(-1, 1)
    coeff_d[:, :, 2].copy_(coeffBernoulli[:, :, 2].clone())

    return torch.cat([coeff_w, coeff_h, coeff_d], dim=2)  # in (bs, nCuboid, 3*nSamplePerFace, 3)


def get_cuboid_corner(dim=5):
    """

    :param dim:
    :return: right most corner tensor in shape (dim**3, 3)
    """
    ticks = torch.arange(-1, 1, 2 / dim)
    xs, ys, zs = torch.meshgrid(*([ticks] * 3))
    return torch.stack([xs, ys, zs], -1).view(-1, 3)


def sample_partial_cuboid(bs, bins, nSamplePerFace):
    """

    :param bs:
    :param bins:
    :param nSamplePerFace:
    :return:
    """
    a = sample_cudoid(bs, bins ** 3, nSamplePerFace)  # bs x bins**3 x 3 * nSamplePerFace x 3
    b = get_cuboid_corner(bins)  # bins**3 x 3
    b = b.unsqueeze(0).repeat(2, 1, 1)  # bs x bins**3 x 3
    return a * (1 / bins) + b.unsqueeze(2)


if __name__ == '__main__':
    batch_size = 2
    bins_per_face = 2
    samples_per_face = 333
    ncuboid = bins_per_face ** 3
    c = sample_partial_cuboid(batch_size, bins_per_face, samples_per_face)
    import numpy.random as rm

    plot_pc_mayavi([c[0][i] for i in range(ncuboid)],
                   colors=[tuple(rm.random(3)) for i in range(ncuboid)])
