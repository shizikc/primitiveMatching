import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.shapeNet import ShapeDiffDataset
from modules.configUtils import get_args, update_tracking, detach_dict
from modules.losses import MatchNetLoss
from modules.matchNet import MatchNet
from tqdm import tqdm

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

torch.backends.cudnn.benchmark = True

run_id = "{:%m%d_%H%M}".format(datetime.now())

params = get_args()

MODEL_LOG_PATH = Path(params.model_path, run_id)

if not os.path.exists(MODEL_LOG_PATH):
    os.makedirs(MODEL_LOG_PATH)

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    filename=Path(MODEL_LOG_PATH, "log.txt"),
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

## Params ##
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

writer = SummaryWriter(MODEL_LOG_PATH)


def get_data():
    train_ds = ShapeDiffDataset(params.train_path, params.bins, dev)
    valid_ds = ShapeDiffDataset(params.val_path, params.bins, dev)
    return (
        DataLoader(train_ds, batch_size=params.batch_size, shuffle=True, drop_last=True),
        DataLoader(valid_ds, batch_size=params.batch_size * 2, drop_last=True)
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
    return model.to(dev), opt, lr_scheduler


def fit(epochs, model, loss_obj, opt, train_dl, valid_dl, lr_opt=None):
    for epoch in tqdm(range(epochs), desc="Epoch", position=0):
        loss_obj.iter = epoch

        model.train()
        loss_obj.reset_loss()

        for x_part, diff_gt, p1_gt, p2_gt in train_dl:
            loss_batch(model, loss_obj.loss_func, diff_gt, (x_part, p1_gt, p2_gt), opt)

        loss_obj.end_epoch(len(train_dl))

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
        writer.add_scalar("Recall (Train)", loss_obj.metrics["recall"], epoch)

        model.eval()
        loss_obj.reset_loss()
        with torch.no_grad():
            for i, (x_part_v, diff_gt_v, p1_gt_v, p2_gt_v) in enumerate(valid_dl):
                loss_batch(model, loss_obj.loss_func, x_part_v, (diff_gt_v, p1_gt_v, p2_gt_v))
            loss_obj.end_epoch(i + 1)

            logging.info(
                "Epoch (Valid): %(epoch).1f, total loss : %(total_loss)5.4f, pred_loss: %(pred_loss).4f,"
                " c_loss: %(c_loss).3f accuracy : %(acc).4f, False negative : %(fn).4f, "
                "Precision : %(precision).4f,  Recall : %(recall).4f" % loss_obj.metrics)

            writer.add_scalar("Loss (Validation)", loss_obj.metrics["total_loss"], epoch)
            writer.add_scalar("Accuracy (Validation)", loss_obj.metrics["acc"], epoch)
            writer.add_scalar("False Negative (Validation)", loss_obj.metrics["fn"], epoch)
            writer.add_scalar("Precision (Validation)", loss_obj.metrics["precision"], epoch)
            writer.add_scalar("Recall (Validation)", loss_obj.metrics["recall"], epoch)

        # TODO: when turning validation - replace minimum loss with val_loss
        if epoch == 0:  # params.reg_start_iter:
            min_loss = loss_obj.metrics['total_loss']

        if epoch >= params.reg_start_iter and loss_obj.metrics['total_loss'] <= min_loss:
            min_loss = loss_obj.metrics['total_loss']

            # save best model
            torch.save(model.state_dict(), Path(MODEL_LOG_PATH, "model.pt"))
    return min_loss


if __name__ == '__main__':
    update_tracking(id=run_id, data=vars(params), csv_file=Path(MODEL_LOG_PATH, "tracking.csv"))

    train_dl, valid_dl = get_data()

    model, opt, opt_rl = get_model()

    MNLoss = MatchNetLoss(threshold=params.threshold, reg_start_iter=params.reg_start_iter,
                          bce_coeff=params.bce_coeff, cd_coeff=params.cd_coeff,
                          fn_coeff=params.fn_coeff, bins=params.bins)
    try:
        min_loss = fit(params.max_epoch, model, MNLoss, opt, train_dl, valid_dl, opt_rl)
    except Exception as e:
        logging.error(e)
        # remove updates folder
        shutil.rmtree(MODEL_LOG_PATH)

    writer.flush()
    writer.close()

    update_tracking(id=run_id, data=detach_dict(MNLoss.metrics),
                    csv_file=Path(MODEL_LOG_PATH, "tracking.csv"))
    update_tracking(id=run_id, data={"min_loss": min_loss.cpu().detach().numpy(),
                                     "end_time": "{:%m%d_%H%M}".format(datetime.now())},
                    csv_file=Path(MODEL_LOG_PATH, "tracking.csv"))
