import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryDSCLoss(nn.Module):
    r"""DSC loss for binary classification.
    
    Args:
        gamma (float, optional): A factor to push down the weight of easy
            examples. Default: 1.0
        smooth (float, optional): a factor added to both the nominator and
            the denominator for smoothing purposes. The smooth parameter.
            Default: 1.0
        reduction (str, optional): The reduction method. Default: 'mean'
    """
    def __init__(self, gamma: float = 1.0, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction

        assert self.reduction in ['mean', 'sum', 'none']

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        probs = torch.sigmoid(logits)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        probs = pos_mask * probs + neg_mask * (1 - probs)

        pos_weight = pos_mask * torch.pow(torch.sub(1.0, probs), self.gamma) * probs
        pos_loss = 1 - (2 * pos_weight + self.smooth) / (pos_weight + 1 + self.smooth)

        neg_weight = neg_mask * torch.pow(torch.sub(1.0, probs), self.gamma) * probs
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
        gamma (float, optional): A factor to push down the weight of easy
            examples. Default: 1.0
        smooth (float, optional): a factor added to both the nominator and
            the denominator for smoothing purposes. The smooth parameter.
            Default: 1.0
        reduction (str, optional): The reduction method. Default: 'mean'
    """
    def __init__(self, gamma: float = 0.0, smooth: float = 1e-4, dice_square: bool = False, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction
        self.dice_square = dice_square

        assert self.reduction in ['mean', 'sum', 'none']

    def _dice_loss(self, flat_input: torch.Tensor, flat_target: torch.Tensor):
        flat_input = ((1 - flat_input) ** self.gamma) * flat_input
        intersection = torch.sum(flat_input * flat_target, dim=-1)
        if not self.dice_square:
            loss = 1 - (2.0 * intersection + self.smooth) / (torch.sum(flat_input, dim=-1) + torch.sum(flat_target, dim=-1) + self.smooth)
        else:
            loss = 1 - (2.0 * intersection + self.smooth) / (torch.sum(flat_input ** 2, dim=-1) + torch.sum(flat_target ** 2, dim=-1) + self.smooth)
        return loss

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        logits_size = logits.shape[-1]
        flat_input = logits
        flat_target = F.one_hot(target, num_classes=logits.size(-1)).float()
        flat_input = torch.softmax(flat_input, dim=-1)

        loss = None
        for label_idx in range(logits_size):
            flat_input_idx = flat_input[:, label_idx]
            flat_target_idx = flat_target[:, label_idx]

            loss_idx = self._dice_loss(flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1))
            if loss is None:
                loss = loss_idx
            else:
                loss += loss_idx

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
