import logging

import torch
import numpy as np
from torch import optim
from torch.utils.data.dataloader import DataLoader

from data.shapeNet import ShapeDiffDataset
from modules.losses import matchNetLoss
from modules.matchNet import MatchNet

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

## Params
bins_per_face = 5
samples_per_face = 333
lr = 0.01
train_path = 'C:/Users/sharon/Documents/Research/data/dataset2019/shapenet/train/gt/03001627'
val_path = 'C:/Users/sharon/Documents/Research/data/dataset2019/shapenet/val/gt/03001627'
batch_size = 2
max_epoch = 2


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_obj, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        loss_obj.iter = epoch
        model.train()
        for x_part, diff_gt, p_gt in train_dl:
            loss_batch(model, loss_obj.loss_func, x_part, (diff_gt, p_gt), opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_obj.loss_func, x_part,
                             (diff_gt, p_gt)) for x_part, diff_gt, p_gt in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        logging.info("epoch: " + str(epoch) + "val_loss: " + str(val_loss))


def get_model():
    model = MatchNet(bins=bins_per_face, samplesPerFace=samples_per_face)
    return model, optim.SGD(model.parameters(), lr=lr)


if __name__ == '__main__':
    train_ds = ShapeDiffDataset(train_path, bins_per_face, dev)
    valid_ds = ShapeDiffDataset(val_path, bins_per_face, dev)

    train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)

    model, opt = get_model()
    MNLoss = matchNetLoss(threshold=0.01, reg_start_iter=0,
                          bce_coeff=1., cd_coeff=1.)
    fit(max_epoch, model, MNLoss, opt, train_dl, valid_dl)
