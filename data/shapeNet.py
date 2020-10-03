import os
import random
from pathlib import Path
import logging
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import numbers

from utils.visualization import plot_pc_mayavi

logging.getLogger().setLevel(logging.INFO)


# PATH = FLAGS.data_path
LOWER_LIM = -1
UPPER_LIM = 1


def load_single_file(path, data_name="data"):
    fx = h5py.File(path, 'r')
    return np.array(fx[data_name])


# UTILITY FUNCTIONS TO CREATE AND SAVE DATASETS#
def create_hist_labels(diff_set, bins):
    """

    :param diff_set: tuple containing (X, Y, Z)
    :return:
    H: ndarray The multidimensional histogram of sample x. See normed and weights for the different possible semantics.
    edges: list A list of D arrays describing the bin edges for each dimension.
    """
    r = (LOWER_LIM, UPPER_LIM)
    H = np.histogramdd((diff_set[:, 0], diff_set[:, 1], diff_set[:, 2]), bins=bins, range=(r, r, r))
    return H[0] / diff_set.shape[0], H[1]


def create_diff_point_cloud(pc1, pc2):
    """
    extracts all points in pc1 but not in pc2
    :param pc1: numpy of shape(num_points, 3) - complete object
    :param pc2: numpy of shape(num_points, 3) - partial object
    :return: numpy of shape(diff num_points, 3)
    """
    _, indices = np.unique(np.concatenate([pc2, pc1]), return_index=True, axis=0)
    indices = indices[indices >= pc2.shape[0]] - pc2.shape[0]
    return pc1[indices]


def create_partial_from_complete(complete, partial_size=0.2, rng=None):
    """
    create a partial object
    The returned object is then added with randomly duplicated points to contain exactly 1740 points
    :param complete:
    :return:
    """
    def ratio2int(r):
        return (np.array(r) * len(complete)).round().astype(np.int32)

    def normalize_size(s):
        s = np.array(s)
        if s.shape not in  [(), (2,)]:
            raise ValueError("Partial size %$ format not supported")
        if not issubclass(s.dtype.type, numbers.Integral):
            s = ratio2int(s)
        return s

    if rng is None:
        rng = np.random.default_rng()

    partial_size = normalize_size(partial_size)
    # project points on a direction, thus creating order
    normal = rng.randn(3)
    normal /= np.linalg.norm(normal)
    proj = complete.dot(normal)
    sort_inds = np.argsort(proj)
    if partial_size.shape == ():
        take = partial_size
    else:
        take = rng.randint(partial_size[0], partial_size[1] + 1)
    partial = complete[sort_inds[:take]]
    diff = complete[sort_inds[take:]]

    # return torch.tensor(partial), torch.tensor(diff)
    return diff, partial


class ShapeDiffDataset(Dataset):
    """shapenet partial dataset"""

    def __init__(self, path, bins, dev, partial_size=256, seed=None):
        """
        path: contains h5 files
        """
        self.path = path
        self.bins = bins
        self.dev = dev
        self.fn_list = os.listdir(self.path)
        self.rng = np.random.RandomState(seed)
        self.partial_size = partial_size

    def __len__(self):
        return len(self.fn_list)

    def __getitem__(self, idx):
        in_path = os.path.join(self.path, self.fn_list[idx])

        x_complete = load_single_file(in_path)
        x_complete *= 2.
        x_partial, x_diff = create_partial_from_complete(x_complete,
                                                            partial_size=self.partial_size,
                                                            rng=self.rng)
        H, edges = create_hist_labels(x_diff, self.bins)

        return torch.tensor(x_partial).to(self.dev).float(), torch.tensor(x_diff).to(self.dev).float(),\
               torch.tensor(H > 0.0).to(self.dev).float()


#
if __name__ == '__main__':
    train_path = 'C:/Users/sharon/Documents/Research/data/dataset2019/shapenet/train/gt/03001627'
    # train_path = '/home/yonatan/data/oc3d/chair/train/gt/03001627'

    shapenet = ShapeDiffDataset(train_path,
                                bins=5,
                                dev='cpu',
                                partial_size=256,
                                seed=42
                                )
    x_partial, x_diff, hist = shapenet[0]
    plot_pc_mayavi([x_partial, x_diff], colors=((1., 1., 1.), (1., 0., 0.)))
