import torch
import numpy as np
from torch import optim
from torch.utils.data.dataloader import DataLoader

from data.shapeNet import ShapeDiffDataset
from modules.losses import loss_func
from modules.matchNet import MatchNet


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


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for x_part, diff_gt, p_gt in train_dl:
            loss_batch(model, loss_func, x_part, (diff_gt, p_gt), opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, x_part, (diff_gt, p_gt)) for x_part, diff_gt, p_gt in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

def get_model():
    model = MatchNet(bins=bins_per_face, samplesPerFace=samples_per_face)
    return model, optim.SGD(model.parameters(), lr=lr)


if __name__ == '__main__':
    train_ds = ShapeDiffDataset(train_path, bins_per_face, dev)
    valid_ds = ShapeDiffDataset(val_path, bins_per_face, dev)

    train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)
    model, opt = get_model()
    fit(max_epoch, model, loss_func, opt, train_dl, valid_dl)
