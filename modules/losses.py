import torch
import torch.nn as nn

bce_loss = nn.BCELoss(reduction='mean')

# TODO: class this and add iteration indicator similar to completion repository
threshold = 0.01
iter = 150  #train cd from first iteration
reg_start_iter = 150
bce_coeff = 1.
cd_coeff = 1.

# TODO: change this to recieve different number of points per sample
def chamfer_distance(a, b, method="mean"):
    """
    a: (b, p, 3)
    b: (b, q, 3)
    """
    diff = a[:, :, None, :] - b[:, None, :, :]
    dist = torch.norm(diff, p=2, dim=3)
    d_min, _ = dist.min(2)
    if method == "mean":
        ch_dist = d_min.mean()
    else:
        ch_dist = d_min.max()
    return ch_dist


def loss_func(pred, gt):
    """

    :param pred: tuple (out, p, z), out in (bs x bins^3 x nSaples x 3),  p in (bs, bins^3),
                                    z in ([bs, bins^3, nSamples, 3])
    :param gt: tuple (hole_gt, p_gt), hole_gt in torch.Size([bs, 256, 3]) and p_gt in torch.Size([bs, bins^3])
    :return:
    """
    prob_target = gt[1]
    probs_pred = pred[1]
    diff_pred = pred[2]

    train_reg = iter >= reg_start_iter

    pred_loss = bce_loss(probs_pred, prob_target)
    mask = probs_pred > threshold

    acc = ((probs_pred > 0.5) == prob_target).float().mean()

    if train_reg:
        if diff_pred.shape[1] > 0:
            CD = chamfer_distance()
    else:
        c_loss = torch.tensor(0.)

    return bce_coeff * pred_loss + cd_coeff * c_loss