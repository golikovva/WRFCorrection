from torch import nn
import torch


class TurbulentMSE(nn.Module):
    def __init__(self, meaner, beta=1, logger=None, kernel_type='mean', k=9):
        super().__init__()
        self.mean_mse = nn.MSELoss()
        self.delta_mse = nn.MSELoss()
        self.kernel_type = kernel_type
        self.delta_conv = nn.Conv2d(3, 3, k, groups=3, bias=False, padding=k // 2, padding_mode='replicate')
        self.delta_conv.weight = self.get_kernel(k)
        self.meaner = meaner
        self.beta = beta
        if logger:
            logger.set_beta(beta)

    def forward(self, orig, corr, target, logger=None):
        mean_corr = self.meaner(corr)

        t = target.view(*target.shape[:-2], target.shape[-1] * target.shape[-2])
        t = t[..., self.meaner.mapping.unique().long()]

        mse1 = self.mean_mse(mean_corr, t)
        # delta_corr = corr.permute(-2, -1, *list(range(len(corr.shape) - 2))) - corr.mean(dim=(-1, -2))
        # delta_orig = orig.permute(-2, -1, *list(range(len(orig.shape) - 2))) - orig.mean(dim=(-1, -2))
        delta_corr = corr - self.delta_conv(corr.view(-1, 3, 210, 280)).view(corr.shape)
        delta_orig = orig - self.delta_conv(orig.view(-1, 3, 210, 280)).view(orig.shape)
        mse2 = self.delta_mse(delta_corr, delta_orig)
        if logger:
            logger.accumulate_stat(mse1 + self.beta * mse2, mse1, mse2)
        return mse1 + self.beta * mse2

    def get_kernel(self, k):
        if self.kernel_type == 'mean':
            w = torch.ones_like(self.delta_conv.weight) / k / k
            return nn.Parameter(w)


class StationMSE(nn.Module):
    def __init__(self, logger=None):
        super().__init__()
        self.station_mse = nn.MSELoss()

    def forward(self, pred, target, logger=None):
        mse1 = self.mean_mse(pred, target)
        if logger:
            logger.accumulate_stat(mse1)
        return mse1



# class TurbulentMSEMetric(nn.Module):
#     def __init__(self, meaner, beta=1., reduction='none'):
#         super().__init__()
#         self.mean_mse = nn.MSELoss(reduction=reduction)
#         self.delta_mse = nn.MSELoss(reduction=reduction)
#         self.meaner = meaner
#         self.beta = beta
#
#     def forward(self, orig, corr, target, logger=None):
#         mean_corr = self.meaner(corr)
#
#         t = target.view(*target.shape[:-2], target.shape[-1] * target.shape[-2])
#         t = t[..., self.meaner.mapping.unique().long()]
# 
#         mse1 = self.mean_mse(mean_corr, t)
#         mse1 = mse1.mean(dim=-1)
#         delta_corr = corr.permute(-2, -1, *list(range(len(corr.shape) - 2))) - corr.mean(dim=(-1, -2))
#         delta_orig = orig.permute(-2, -1, *list(range(len(orig.shape) - 2))) - orig.mean(dim=(-1, -2))
#         mse2 = self.delta_mse(delta_corr, delta_orig)
#         mse2 = mse2.mean(dim=(0, 1))
#
#         return mse1 + self.beta * mse2
