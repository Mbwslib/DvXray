from torch import nn
import torch

class BCELoss(torch.nn.Module):
    def __init__(self, reduction='sum'):
        super(BCELoss, self).__init__()

        self.reduction = reduction
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, ol_output, sd_output, gt_s):

        ol_bce_loss = self.loss_fct(ol_output, gt_s)

        sd_bce_loss = self.loss_fct(sd_output, gt_s)

        if self.reduction == 'mean':
            loss = torch.mean(ol_bce_loss + sd_bce_loss)
        elif self.reduction == 'sum':
            loss = torch.sum(ol_bce_loss + sd_bce_loss)
        return loss