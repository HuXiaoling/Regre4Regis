from utilities import SoftDiceLoss, softmax_helper
import torch.nn as nn

soft_dice_args = {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}
ce_loss = nn.CrossEntropyLoss()

sdl = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_args)
l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()

## TO DO: Add more losses here
