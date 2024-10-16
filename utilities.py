# loss functions
# apply softmax on network output for dice, not CE
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np


softmax_helper = lambda x: F.softmax(x, 1)

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

        # print("batch_dice: {}\ndo_bg: {}\n".format(self.batch_dice, self.do_bg))

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        #print("[SAUMDEBUG]\naxes: {}\n".format(axes))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
            #print("[SAUMDEBUG]\napply_nonlin called")

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()
        #print("[SAUMDEBUG]\ndc without manipulation: {}\n -dc: {}\n".format(dc, -dc))
        return -dc

class IOU(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(IOU, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

        # print("batch_dice: {}\ndo_bg: {}\n".format(self.batch_dice, self.do_bg))

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        #print("[SAUMDEBUG]\naxes: {}\n".format(axes))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
            #print("[SAUMDEBUG]\napply_nonlin called")

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = tp + self.smooth
        denominator = tp + fp + fn + self.smooth

        iou = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                iou = iou[1:]
            else:
                diouc = iou[:, 1:]
        iou = iou.mean()
        #print("[SAUMDEBUG]\ndc without manipulation: {}\n -dc: {}\n".format(dc, -dc))
        return iou

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label

        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
            #print("[SAUMDEBUG]\nDC_and_CE_loss\nweight_ce: {}\nce_loss: {}\nweight_dice: {}\ndc_loss: {}\n".format(self.weight_ce, ce_loss, self.weight_dice, dc_loss))
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

def dice_loss(input, target):
    smooth = 1.

    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def onehot_encoding(seg, onehotmatrix, lut):
    # one hot encoding
    label = seg[:,0,:]
    seg_onehot = onehotmatrix[lut[label.long()]]
    seg_onehot = seg_onehot.permute(0, 4, 1, 2, 3)

    return seg_onehot

def uncer_loss_single_gaussian(y_pred, y_gt, mask):
    uncer_loss = 0.5 * torch.mean(y_pred[:,3,] + (y_pred[:,0,] - y_gt[:,0,:]) ** 2/(1e-3 * torch.exp(y_pred[:,3,])) + \
                                                 (y_pred[:,1,] - y_gt[:,1,:]) ** 2/(1e-3 * torch.exp(y_pred[:,3,])) + \
                                                 (y_pred[:,2,] - y_gt[:,2,:]) ** 2/(1e-3 * torch.exp(y_pred[:,3,]))) / (1e-6 + torch.mean(mask))
    return uncer_loss

def uncer_loss_single_lap(y_pred, y_gt, mask):
    l1_loss = nn.L1Loss()
    uncer_loss = torch.mean(y_pred[:,3,] + l1_loss(y_pred[:,0,] / (0.03 * torch.exp(y_pred[:,3,])), y_gt[:,0,:] * mask / (0.03 * torch.exp(y_pred[:,3,]))) + \
                                           l1_loss(y_pred[:,1,] / (0.03 * torch.exp(y_pred[:,3,])), y_gt[:,1,:] / (0.03 * torch.exp(y_pred[:,3,]))) + \
                                           l1_loss(y_pred[:,2,] / (0.03 * torch.exp(y_pred[:,3,])), y_gt[:,2,:] / (0.03 * torch.exp(y_pred[:,3,])))) / (1e-6 + torch.mean(mask))
    return uncer_loss

def uncer_loss_three_gaussian(y_pred, y_gt, mask):
    uncer_loss = 0.5 * torch.mean(y_pred[:,3,] + (y_pred[:,0,] - y_gt[:,0,:]) ** 2/(1e-3 * torch.exp(y_pred[:,3,])) + \
                                  y_pred[:,4,] + (y_pred[:,1,] - y_gt[:,1,:]) ** 2/(1e-3 * torch.exp(y_pred[:,4,])) + \
                                  y_pred[:,5,] + (y_pred[:,2,] - y_gt[:,2,:]) ** 2/(1e-3 * torch.exp(y_pred[:,5,]))) / (1e-6 + torch.mean(mask))
    return uncer_loss

def uncer_loss_three_lap(y_pred, y_gt, mask):
    l1_loss = nn.L1Loss()
    uncer_loss = torch.mean(y_pred[:,3,] + l1_loss(y_pred[:,0,] / (0.03 * torch.exp(y_pred[:,3,])), y_gt[:,0,:] / (0.03 * torch.exp(y_pred[:,3,]))) + \
                            y_pred[:,4,] + l1_loss(y_pred[:,1,] / (0.03 * torch.exp(y_pred[:,4,])), y_gt[:,1,:] / (0.03 * torch.exp(y_pred[:,4,]))) + \
                            y_pred[:,5,] + l1_loss(y_pred[:,2,] / (0.03 * torch.exp(y_pred[:,5,])), y_gt[:,2,:] / (0.03 * torch.exp(y_pred[:,5,])))) / (1e-6 + torch.mean(mask))
    return uncer_loss