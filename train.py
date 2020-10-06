import logging
from datetime import datetime

import torch
import numpy as np
from torch import optim
from torch.utils.data.dataloader import DataLoader

from data.shapeNet import ShapeDiffDataset
from modules.configUtils import get_args, update_tracking
from modules.losses import matchNetLoss
from modules.matchNet import MatchNet

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

run_id = "{:%m%d_%H%M}".format(datetime.now())

## Params ##

params = get_args()


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

        # model.eval()
        # with torch.no_grad():
        #     losses, nums = zip(
        #         *[loss_batch(model, loss_obj.loss_func, x_part,
        #                      (diff_gt, p_gt)) for x_part, diff_gt, p_gt in valid_dl]
        #     )
        # val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        #
        # logging.info("Epoch (Valid): {:3d}, total loss : {:05.4f}".format(epoch, val_loss))
        # TODO: when turning validation - replace minimum loss with val_loss
        if epoch == params.reg_start_iter:
            min_loss = loss_obj.temp_metrics['total_loss']

        if epoch >= params.reg_start_iter and loss_obj.temp_metrics['total_loss'] <= min_loss:
            min_loss = loss_obj.temp_metrics['total_loss']

            # save minimum model
            torch.save(model.state_dict(), params.model_path + "model_" + str(run_id) + ".pt")


def get_model():
    model = MatchNet(bins=params.bins_per_face, samplesPerFace=params.samples_per_face, dev=dev)
    if params.optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr=params.lr)
    else:
        opt = optim.SGD(model.parameters(), lr=params.lr, momentum=params.mmnt)
    return model.to(dev), opt


update_tracking(run_id, "optimizer", params.optimizer)
update_tracking(run_id, "bins", params.bins)
update_tracking(run_id, "samples_per_face", params.samples_per_face)
update_tracking(run_id, "lr", params.lr)
update_tracking(run_id, "momentum", params.momentum)
update_tracking(run_id, "train_path", params.train_path)
update_tracking(run_id, "val_path", params.val_path)
update_tracking(run_id, "batch_size", params.batch_size)
update_tracking(run_id, "max_epoch", params.max_epoch)
update_tracking(run_id, "threshold", params.threshold)
update_tracking(run_id, "reg_start_iter", params.reg_start_iter)
update_tracking(run_id, "bce_coeff", params.bce_coeff)
update_tracking(run_id, "cd_coeff", params.cd_coeff)


if __name__ == '__main__':
    train_ds = ShapeDiffDataset(params.train_path, params.bins, dev)
    valid_ds = ShapeDiffDataset(params.val_path, params.bins, dev)

    train_dl, valid_dl = get_data(train_ds, valid_ds, params.batch_size)

    model, opt = get_model()
    MNLoss = matchNetLoss(threshold=params.threshold, reg_start_iter=params.reg_start_iter,
                          bce_coeff=params.bce_coeff, cd_coeff=params.cd_coeff)
    fit(params.max_epoch, model, MNLoss, opt, train_dl, valid_dl)
