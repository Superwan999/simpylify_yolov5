import torch
import torch.nn as nn
from utils.torch_utils import *
from utils.metrics import bbox_iou


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps

class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class ComputeLoss:
    sort_obj_iou = False

    def __init__(self, model, autobalance=False):
        device = model.device
        hyp = model.hyp
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['obj_pw']], device=device))

        self.cp, self.cn = smooth_BCE(eps=hyp.get('label_smoothing', 0.0))
        g = hyp['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m_detect = de_parallel(model).model[-1]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m_detect.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(m_detect).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, hyp, autobalance
        self.na = m_detect.na
        self.nc = m_detect.nc
        self.nl = m_detect.nl
        self.anchors = m_detect.anchors
        self.device = device

    def __call__(self, p, targets):
        cls_loss = torch.zeros(1, device=self.device)
        box_loss = torch.zeros(1, device=self.device)
        obj_loss = torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i] # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)

            n = b.shape[0]
            if n:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbox[i], iou_type=self.hyp['iou_type']).squeeze()
                box_loss += (1 - iou).mean()

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]

                if self.gr < 1:
                    iou = (1 - self.gr) + self.gr * iou

                tobj[b, a, gj, gi] = iou

                # Classification
                if self.nc > 1:
                    t = torch.full_like(pcls, self.cn, device=self.device)
                    t[range(n), tcls[i]] = self.cp
                    cls_loss += self.BCEcls(pcls, t)

            obji = self.BCEobj(pi[..., 4], tobj)
            obj_loss += obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        box_loss *= self.hyp['loss']['box']
        obj_loss *= self.hyp['loss']['obj']
        cls_loss *= self.hyp['loss']['cls']
        bs = targets.shape[0]

        return (box_loss + obj_loss + cls_loss) * bs, torch.cat((box_loss, obj_loss, cls_loss)).detach()

    def build_targets(self, p, targets):
        na, nt = self.na, targets.shape[0] # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)

        g = 0.5

        # offset
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
            ],
            device=self.device
        ).float() * g

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Match targets to  anchors
                r = t[..., 4:6] / anchors[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']
                t = t[j]

                # offsets
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offset = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offset = 0

            # define
            bc, gxy, gwh, a = t.chunk(4, 1)
            a, (b, c) = a.long().view(-1), bc.long().T
            gij = (gxy - offset).long()
            gi, gj = gij.T

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)
        return tcls, tbox, indices, anch






