import logging
from datetime import datetime
import torch
import numpy as np
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.shapeNet import ShapeDiffDataset
from modules.configUtils import get_args, update_tracking
from modules.losses import MatchNetLoss
from modules.matchNet import MatchNet

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

## Params ##
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

run_id = "{:%m%d_%H%M}".format(datetime.now())

params = get_args()

writer = SummaryWriter(params.log_dir)


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True),
        DataLoader(valid_ds, batch_size=bs * 2, drop_last=True),
    )


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), len(xb)


def get_model():
    model = MatchNet(bins=params.bins, samplesPerFace=params.samples_per_face, dev=dev)

    if params.optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr=params.lr)
    else:
        opt = optim.SGD(model.parameters(),
                        lr=params.lr,
                        momentum=params.momentum)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt,
                                                   step_size=params.step_size,
                                                   gamma=params.gamma)
    return model.to(dev), (opt, lr_scheduler)


def fit(epochs, model, loss_obj, opt, train_dl, valid_dl, lr_opt=None):

    for epoch in range(epochs):
        loss_obj.iter = epoch

        # if epoch > 0:
        #     print("hi")

        model.train()
        loss_obj.reset_loss()

        for idx, (x_part, diff_gt, p_gt) in enumerate(train_dl):
            loss_batch(model, loss_obj.loss_func, diff_gt, (x_part, p_gt), opt)

        loss_obj.end_epoch(idx + 1)

        logging.info(
            "Epoch (Train): %(epoch).1f, total loss : %(total_loss)5.4f, pred_loss: %(pred_loss).4f,"
            " c_loss: %(c_loss).3f accuracy : %(acc).4f, False negative : %(fn).4f, "
            "Precision : %(precision).4f,  Recall : %(recall).4f" % loss_obj.metrics)

        # update the learning rate
        if lr_opt.get_last_lr()[0] > params.min_lr:
            lr_opt.step()

        writer.add_scalar("Loss (Train)", loss_obj.metrics["total_loss"], epoch)
        writer.add_scalar("Accuracy (Train)", loss_obj.metrics["acc"], epoch)
        writer.add_scalar("False Negative (Train)", loss_obj.metrics["fn"], epoch)
        writer.add_scalar("Precision (Train)", loss_obj.metrics["precision"], epoch)
        writer.add_scalar("Recall", loss_obj.metrics["recall"], epoch)

        model.eval()
        loss_obj.reset_loss()
        with torch.no_grad():
            for i, (x_part_v, diff_gt_v, p_gt_v) in enumerate(valid_dl):
                loss_batch(model, loss_obj.loss_func, x_part_v, (diff_gt_v, p_gt_v))
            loss_obj.end_epoch(i + 1)

            logging.info(
                "Epoch (Valid): %(epoch).1f, total loss : %(total_loss)5.4f, pred_loss: %(pred_loss).4f,"
                " c_loss: %(c_loss).3f accuracy : %(acc).4f, False negative : %(fn).4f, "
                "Precision : %(precision).4f,  Recall : %(recall).4f" % loss_obj.metrics)

            writer.add_scalar("Loss (Validation)", loss_obj.metrics["total_loss"], epoch)
            writer.add_scalar("Accuracy (Validation)", loss_obj.metrics["acc"], epoch)
            writer.add_scalar("False Negative (Validation)", loss_obj.metrics["fn"], epoch)

        # # TODO: when turning validation - replace minimum loss with val_loss
        # if epoch == params.reg_start_iter:
        #     min_loss = loss_obj.metrics['total_loss']
        #
        # if epoch >= params.reg_start_iter and loss_obj.metrics['total_loss'] <= min_loss:
        #     min_loss = loss_obj.metrics['total_loss']
        #
        #     # save minimum model
        #     torch.save(model.state_dict(), params.model_path + "model_" + str(run_id) + ".pt")


if __name__ == '__main__':
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
    update_tracking(run_id, "fn_coeff", params.fn_coeff)

    train_ds = ShapeDiffDataset(params.train_path, params.bins, dev)
    valid_ds = ShapeDiffDataset(params.val_path, params.bins, dev)

    train_dl, valid_dl = get_data(train_ds, valid_ds, params.batch_size)

    model, opt = get_model()

    MNLoss = MatchNetLoss(threshold=params.threshold, reg_start_iter=params.reg_start_iter,
                          bce_coeff=params.bce_coeff, cd_coeff=params.cd_coeff, fn_coeff=params.fn_coeff, bins=params.bins)

    fit(params.max_epoch, model, MNLoss, opt[0], train_dl, valid_dl, opt[1])

    writer.flush()
    writer.close()

    update_tracking(run_id, "total_loss", MNLoss.metrics["total_loss"].cpu().detach().numpy())
    update_tracking(run_id, "pred_loss", MNLoss.metrics["pred_loss"].cpu().detach().numpy())
    update_tracking(run_id, "c_loss", MNLoss.metrics["c_loss"].cpu().detach().numpy())
    update_tracking(run_id, "Accuracy", MNLoss.metrics["acc"].cpu().detach().numpy())
    update_tracking(run_id, "ended_time", "{:%m%d_%H%M}".format(datetime.now()))
