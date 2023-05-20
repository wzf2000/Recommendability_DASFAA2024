import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryDSCLoss(nn.Module):
    r"""DSC loss for binary classification.
    
    Args:
        alpha (float, optional): A factor to push down the weight of easy
            examples. Default: 1.0
        smooth (float, optional): a factor added to both the nominator and
            the denominator for smoothing purposes. The smooth parameter.
            Default: 1.0
        reduction (str, optional): The reduction method. Default: 'mean'
    """
    def __init__(self, alpha: float = 1.0, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.reduction = reduction

        assert self.reduction in ['mean', 'sum', 'none']

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        probs = torch.sigmoid(logits)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        probs = pos_mask * probs + neg_mask * (1 - probs)

        pos_weight = pos_mask * torch.pow(torch.sub(1.0, probs), self.alpha) * probs
        pos_loss = 1 - (2 * pos_weight + self.smooth) / (pos_weight + 1 + self.smooth)

        neg_weight = neg_mask * torch.pow(torch.sub(1.0, probs), self.alpha) * probs
        neg_loss = 1 - (2 * neg_weight + self.smooth) / (neg_weight + self.smooth)

        loss = pos_loss + neg_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class MultiDSCLoss(nn.Module):
    r"""DSC loss for multi-classification.

    Args:
        num_class (int): The number of classes
        alpha (float, optional): A factor to push down the weight of easy
            examples. Default: 1.0
        smooth (float, optional): a factor added to both the nominator and
            the denominator for smoothing purposes. The smooth parameter.
            Default: 1.0
        reduction (str, optional): The reduction method. Default: 'mean'
    """
    def __init__(self, alpha: float = 1.0, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.reduction = reduction

        assert self.reduction in ['mean', 'sum', 'none']

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        probs = torch.softmax(logits, dim=1)
        probs = probs.gather(dim=1, index=target.unsqueeze(1))

        probs_with_factor = torch.pow((1 - probs), self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.smooth) / (probs_with_factor + 1 + self.smooth)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
