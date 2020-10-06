import torch
import torch.nn.functional as F
# from utils.visualization import plot_pc_mayavi

#
# def cuboidAreaModule(self, dims):
#     """
#     :param: dims:
#     :return: tensor in shape (1, self.nSamples, 1) where dim 2 is cuboid area
#     """
#
#     width, height, depth = torch.chunk(dims, chunks=3, dim=2)
#
#     wh = width * height
#     hd = height * depth
#     wd = width * depth
#
#     surfArea = 2 * (wh + hd + wd)
#     areaRep = surfArea.repeat(1, self.nSamples, 1)
#     return areaRep
#
# def sample_wt_module(self, dims):
#     # dims is bs x 1 x 3
#     area = self.cuboidAreaModule(dims)  # bs x 1 x 1
#     dimsInv = dims.pow(-1)
#     dimsInvNorm = dimsInv.sum(2).repeat(1, 1, 3)
#     normWeights = 3 * (dimsInv / dimsInvNorm)
#
#     widthWt, heightWt, depthWt = torch.chunk(normWeights, chunks=3, dim=2)
#     widthWt = widthWt.repeat(1, self.samplesPerFace, 1)
#     heightWt = heightWt.repeat(1, self.samplesPerFace, 1)
#     depthWt = depthWt.repeat(1, self.samplesPerFace, 1)
#
#     sampleWt = torch.cat([widthWt, heightWt, depthWt], dim=1)
#     finalWt = (1 / self.samplesPerFace) * (sampleWt * area)
#     return finalWt

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


# def sample_partial_cuboid(bs, bins, nSamplePerFace):
#     """
#
#     :param bs:
#     :param bins:
#     :param nSamplePerFace:
#     :return:
#     """
#     a = sample_cudoid(bs, bins ** 3, nSamplePerFace)  # bs x bins**3 x 3 * nSamplePerFace x 3
#     b = get_cuboid_corner(bins)  # bins**3 x 3
#     b = b.unsqueeze(0).repeat(bs, 1, 1)  # bs x bins**3 x 3
#     return a * (1 / bins) + b.unsqueeze(2)




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





def rotate_cuboid(quats):
    """

    :param quats: in (bs, nCubiods, 4) assumes normelized vector
    :return: (bs, nCubiods, 3, 3)
    """
    a = quats[:, :, 0]  # bs x nCubiods
    b = quats[:, :, 1]
    c = quats[:, :, 2]
    d = quats[:, :, 3]
    q2 = torch.square(quats)
    a2 = q2[:, :, 0]  # bs x nCubiods
    b2 = q2[:, :, 1]
    c2 = q2[:, :, 2]
    d2 = q2[:, :, 3]
    ab = a * b  # bs x nCubiods
    ac = a * c
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d

    rot_flat = torch.stack([
        a2 + b2 - c2 - d2, 2 * (bc - ad), 2 * (bd + ac),
        2 * (bc + ad), a2 + c2 - b2 - d2, 2 * (cd - ab),
        2 * (bd - ac), 2 * (cd + ab), a2 + d2 - b2 - c2], dim=2)  # bs x nCubiods x 9

    return rot_flat.reshape(quats.shape[0], quats.shape[1], 3, 3)


if __name__ == '__main__':
    import numpy.random as rm

    batch_size = 2
    bins_per_face = 1
    samples_per_face = 333
    ncuboid = bins_per_face ** 3

    a = sample_cudoid(batch_size, bins_per_face ** 3, samples_per_face)

    k = torch.rand(batch_size, ncuboid, 4)

    # rotate a by k

    k = F.normalize(k, p=2, dim=2)
    k = rotate_cuboid(k) # (bs, nCubiods, 3, 3)
    k = k.unsqueeze(2)

    # rotate by k
    a2 = a.unsqueeze(4)
    a2 = torch.matmul(k, a2).squeeze(4)

    plot_pc_mayavi([a[0][0], a2[0][0]], colors=[tuple(rm.random(3)) for i in range(2)])

    # move a to cuboids grid corners
    b = get_cuboid_corner(bins_per_face)  # bins**3 x 3
    b = b.unsqueeze(0).repeat(batch_size, 1, 1)  # bs x bins**3 x 3
    c = a * (1 / bins_per_face) + b.unsqueeze(2)

    # plot_pc_mayavi([c[0][i] for i in range(ncuboid)],
    #                colors=[tuple(rm.random(3)) for i in range(ncuboid)])
