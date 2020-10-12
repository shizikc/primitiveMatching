import torch
import torch.nn as nn

from modules.cuboid import get_voxel_centers


class MatchNetLoss(nn.Module):
    def __init__(self, threshold=0.01, iter=0, reg_start_iter=150,
                 bce_coeff=1., cd_coeff=1., bins=5):
        super(MatchNetLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.threshold = threshold  # 0.01
        self.iter = iter  # 150  # train cd from first iteration
        self.reg_start_iter = reg_start_iter  # 150
        self.bce_coeff = bce_coeff
        self.cd_coeff = cd_coeff
        self.metrics = None
        self.centers = get_voxel_centers(bins)  # bins^3 x 3

    def loss_func(self, pred, gt):
        """

        :param pred: tuple (out, p, z):
                        out in (bs x bins^3 x nSaples x 3) or None
                        p in (bs, bins^3),
                        z in ([bs, bins^3, nSamples, 3]) or None
        :param gt: tuple (hole_gt, p_gt):
                        hole_gt in torch.Size([bs, 256, 3])
                        p_gt in torch.Size([bs, bins^3])
        :return:
        """
        diff_gt = gt[0] # bs x 256 x  3
        prob_target = gt[1]

        if pred[0] is not None:
            diff_pred = pred[0]  # bs x bins^3 x nSaples x 3
        probs_pred = pred[1]  # bs x bins^3

        dev = prob_target.device
        self.centers = self.centers.to(dev)

        bs = diff_gt.shape[0]

        train_reg = self.iter >= self.reg_start_iter
        # entroy loss
        pred_loss = self.bce_loss(probs_pred, prob_target)

        mask = probs_pred > self.threshold  # bs x bins^3

        acc = (mask == prob_target).float().mean()

        fn = ((mask != prob_target).float() * (prob_target == 1).float()).mean()

        # mask = mask.unsqueeze(2).repeat(1, 1, 3)  # torch.Size([bs, bins^3, 3])
        # mask = mask.unsqueeze(2).repeat(1, 1, diff_pred.shape[2], 1)

        if train_reg:
            # TODO: replace loop
            CD = 0.
            for i in range(bs):
                # diff_pred_i = diff_pred[i][mask[i]].reshape(-1, 3)  # nPoints predicted x 3
                pred_i = self.centers[mask[i], :]  # torch.Size([nPositiveBins, 3])
                # any points detected
                if pred_i.shape[0] > 0:
                    CD += chamfer_distance(pred_i.unsqueeze(0), diff_gt[i].unsqueeze(0))
                    # CD += symmetric_chamfer_distance(pred_i.unsqueeze(0), diff_gt[i].unsqueeze(0))
            c_loss = CD / bs
        else:
            c_loss = torch.tensor(0.)

        total_loss = self.bce_coeff * pred_loss + self.cd_coeff * c_loss + fn # + acc

        self.metrics = {'epoch': self.iter,
                        'total_loss': total_loss,
                        'pred_loss': pred_loss,
                        'c_loss': c_loss,
                        'acc': acc,
                        "fn": fn}

        return total_loss


def symmetric_chamfer_distance(a, b, method="mean"):
    CD1 = chamfer_distance(a, b, method)
    CD2 = chamfer_distance(b, a, method)
    return torch.max(CD1, CD2)


def chamfer_distance(a, b, method="mean"):
    """
    a: (b, p, 3)
    b: (b, q, 3)
    """
    diff = a[:, :, None, :] - b[:, None, :, :]  # (b, p, q, 3)
    dist = diff.norm(p=2, dim=3)
    d_min, _ = dist.min(2)
    if method == "mean":
        ch_dist = d_min.mean()
    else:
        ch_dist = d_min.max()
    return ch_dist
