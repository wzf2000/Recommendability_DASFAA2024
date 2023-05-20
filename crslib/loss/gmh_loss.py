import torch
import torch.nn as nn
import torch.nn.functional as F

class GMHLoss(nn.Module):
    r"""Generalized Mean Huber Loss
    
    Args:
        bins (int, optional): The number of bins. Default: 10
        alpha (float, optional): The alpha parameter. Default: 0.5
    """
    def __init__(self, bins=10, alpha=0.5):
        super(GMHLoss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 1e-4)).long()
    
    def _custom_loss(self, x, target, weight):
        raise NotImplementedError
    
    def _custom_loss_grad(self, x, target):
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.bincount(input=bin_idx, minlength=self._bins).float()
        
        if x.dim() == 2:
            N = (x.shape[0] * x.shape[1])
        else:
            N = x.shape[0]

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = torch.sum(bin_count > 0).item()
        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=1e-4)
        beta = N / gd

        return self._custom_loss(x, target, weight=beta[bin_idx])
    
class GMHCLoss(GMHLoss):
    r"""Generalized Mean Huber Classification Loss
    """
    def __init__(self, bins=10, alpha=0.5):
        super(GMHCLoss, self).__init__(bins, alpha)

    def _custom_loss(self, x: torch.Tensor, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight, reduction='mean')
    
    def _custom_loss_grad(self, x: torch.Tensor, target):
        return torch.sigmoid(x).detach() - target
    
class GMHRLoss(GMHLoss):
    r"""Generalized Mean Huber Regression Loss
    """
    def __init__(self, mu=0, bins=10, alpha=0.5):
        super(GMHRLoss, self).__init__(bins, alpha)
        self._mu = mu
    
    def _custom_loss(self, x: torch.Tensor, target, weight):
        d = x - target
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        if x.dim() == 2:
            N = (x.shape[0] * x.shape[1])
        else:
            N = x.shape[0]
        return torch.sum(loss * weight) / N

    def _custom_loss_grad(self, x: torch.Tensor, target):
        d = x - target
        mu = self._mu
        return d / torch.sqrt(d * d + mu * mu)
