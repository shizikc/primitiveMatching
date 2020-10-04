import logging

import torch
import numpy as np
from torch import optim
from torch.utils.data.dataloader import DataLoader

from data.shapeNet import ShapeDiffDataset
from modules.configUtils import get_args
from modules.losses import matchNetLoss
from modules.matchNet import MatchNet

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

## Params ##

params = get_args()

bins_per_face = params.bins
samples_per_face = params.samples_per_face
lr = params.lr
mmnt = params.momentum
train_path = params.train_path
val_path = params.val_path
model_path = params.model_path
batch_size = params.batch_size
max_epoch = params.max_epoch
threshold = params.threshold
reg_start_iter = params.reg_start_iter
bce_coeff = params.bce_coeff
cd_coeff = params.cd_coeff


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

        logging.info(
            "Epoch (Train): %(epoch)3d, total loss : %(total_loss)5.4f, pred_loss: %(pred_loss).4f,"
            " c_loss: %(c_loss).3f accuracy : %(acc).4f" % loss_obj.temp_metrics)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_obj.loss_func, x_part,
                             (diff_gt, p_gt)) for x_part, diff_gt, p_gt in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        logging.info("Epoch (Valid): {:3d}, total loss : {:05.4f}".format(epoch, val_loss))

        if epoch == reg_start_iter:
            min_loss = val_loss

        if epoch >= reg_start_iter and val_loss <= min_loss:
            min_loss = val_loss

            # save minimum model
            torch.save(model.state_dict(), model_path)


def get_model():
    model = MatchNet(bins=bins_per_face, samplesPerFace=samples_per_face, dev=dev)
    return model.to(dev), optim.SGD(model.parameters(), lr=lr, momentum=mmnt)


if __name__ == '__main__':
    train_ds = ShapeDiffDataset(train_path, bins_per_face, dev)
    valid_ds = ShapeDiffDataset(val_path, bins_per_face, dev)

    train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)

    model, opt = get_model()
    MNLoss = matchNetLoss(threshold=threshold, reg_start_iter=reg_start_iter,
                          bce_coeff=bce_coeff, cd_coeff=cd_coeff)
    fit(max_epoch, model, MNLoss, opt, train_dl, valid_dl)
