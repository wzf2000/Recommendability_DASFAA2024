import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BinaryFocalLoss(nn.Module):
    r"""Focal loss for binary classification.

    Focal_Loss= -1 * alpha * (1 - pt) * log(pt)

    Args:
        alpha (int, float, optional): 3D or 4D the scalar factor for this criterion. Default: 1
        gamma (float, double, optional): gamma > 0 reduces the relative loss for well-classified
            examples (p > 0.5) putting more focus on hard misclassified example. Default: 2
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-6

        assert self.reduction in ['mean', 'sum', 'none']

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        target = target.unsqueeze(dim=1)
        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_weight = (pos_mask * torch.pow(torch.sub(1.0, prob), self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * torch.log(torch.sub(1.0, prob))

        loss = pos_loss + neg_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class MultiFocalLoss(nn.Module):
    r"""Forward focal loss for multi-classification.
    
    Focal_Loss = -1 * alpha * ((1 - pt) ** gamma) * log(pt)

    Args:
        num_class (int): The number of classes
        alpha (int, float, list, tuple, np.ndarray, torch.Tensor, optional): 3D or 4D the scalar
            factor for this criterion. Default: None
        gamma (float, double, optional): gamma > 0 reduces the relative loss for well-classified
            examples (p > 0.5) putting more focus on hard misclassified example. Default: 2
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``
    """
    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        if alpha == None:
            self.alpha = torch.ones(self.num_class)
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, tuple, np.ndarray)):
            assert len(alpha) == num_class
            self.alpha = torch.as_tensor(alpha)
        elif isinstance(alpha, torch.Tensor):
            assert len(alpha) == num_class
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type, expect None, int, float, list, tuple, np.ndarray or torch.Tensor')
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4

        assert self.reduction in ['mean', 'sum', 'none']

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        target = target.view(-1, 1)
        pt = torch.softmax(logits, dim=-1)
        logpt = torch.log_softmax(logits, dim=-1)
        pt = pt.gather(1, target).squeeze(-1)
        logpt = logpt.gather(1, target).squeeze(-1)

        alpha = self.alpha.to(logits.device)
        at = alpha.gather(0, target.squeeze(-1))
        logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if torch.isfinite(loss).logical_not().long().sum() != 0:
            while True:
                test = input('>>> ')
                if test != 'q':
                    try:
                        exec(f'print({test})')
                    except Exception as e:
                        print(str(e))
                else:
                    break

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()


        return loss