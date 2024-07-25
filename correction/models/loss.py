import time
import matplotlib.pyplot as plt
from torch import nn
import torch
import numpy as np
import pendulum as pdl
# from correction.config.config import cfg
from correction.models.model_parts import DeltaConvLayer
from correction.helpers.interpolation import interpolate_input_to_scat


class TurbulentMSE(nn.Module):
    def __init__(self, meaner, betas, station_interpolator=None, scatter_interpolator=None, logger=None,
                 kernel_type='gauss', k=9, device='cpu'):
        super().__init__()
        self.mean_mse = nn.MSELoss()
        self.delta_mse = nn.MSELoss()
        self.station_mse = nn.MSELoss(reduction='none')
        self.scat_mse = nn.MSELoss()
        self.delta_conv = DeltaConvLayer(k, kernel_type=kernel_type)
        self.station_interpolator = station_interpolator
        self.scatter_interpolator = scatter_interpolator
        if scatter_interpolator:
            self.wrf_mask = scatter_interpolator.calc_input_tensor_mask([132, 430], fill_value=0)
        self.meaner = meaner
        self.betas = betas
        self.betas[2] = torch.tensor(self.betas[2], device=device)

        if logger:
            logger.set_beta(betas)

    def forward(self, orig, corr, target=None, stations=None, scatter=None, i=None, start_date=None,
                orig_scaler=None, logger=None, expanded_out=False):
        device = orig.device
        mse1 = torch.zeros(1, device=device)
        if target is not None and self.betas[0] > 0:
            mean_corr, t = self.correlate_wrf_era(corr, target)
            mse1 = self.mean_mse(mean_corr, t)

        delta_corr = corr - self.delta_conv(corr.view(-1, 3, 210, 280)).view(corr.shape)
        delta_orig = orig - self.delta_conv(orig.view(-1, 3, 210, 280)).view(orig.shape)
        mse2 = self.delta_mse(delta_corr, delta_orig)

        mse3 = torch.zeros([1, 1, 2, 1], device=device)
        if stations is not None and self.betas[2].sum() > 0:
            assert orig_scaler is not None, f'WRF scaler needed to calculate station loss'
            pred_stations = self.station_interpolator(orig_scaler.inverse_transform(corr,
                                                                                    means=orig_scaler.channel_means[:3],
                                                                                    stds=orig_scaler.channel_stddevs[:3],
                                                                                    dims=2).flatten(-2, -1))
            pred_stations = uvt_to_wt(pred_stations)
            stations[torch.where(stations.isnan())] = pred_stations[torch.where(stations.isnan())].clone()
            mse3 = self.station_mse(pred_stations, stations)
        mse4 = torch.zeros(1, device=device)
        if scatter is not None and self.betas[3] > 0:
            wrf_on_scat_grid, scat = interpolate_input_to_scat(corr[..., :2, :, :], scatter,
                                                               self.scatter_interpolator, i,
                                                               start_date, self.wrf_mask)
            mse4 = self.scat_mse(wrf_on_scat_grid, scat)

        total_mse = self.betas[0] * mse1 + self.betas[1] * mse2 + torch.einsum('sbcn,c->sbcn', mse3, self.betas[2]).mean() \
                    + self.betas[3] * mse4
        if logger:
            logger.accumulate_stat(total_mse.item(), mse1.item(), mse2.item(),
                                   torch.einsum('sbcn,c->sbcn', mse3, self.betas[2]).mean().item(), mse4.item())
        if expanded_out:
            return total_mse, mse1, mse2, mse3, mse4
        return total_mse

    def correlate_wrf_era(self, corr, target):
        mean_corr = self.meaner(corr)
        t = target.flatten(-2, -1)
        t = t[..., self.meaner.mapping.unique().long()]
        return mean_corr, t


def uvt_to_wt(pred_st, c_dim=-2):
    assert pred_st.shape[c_dim] == 3, f'assumed 3 channels (u, v, t) to be processed but got {pred_st.shape[c_dim]}'
    u, v, t = torch.split(pred_st, 1, dim=c_dim)
    w = torch.sqrt(torch.square(u) + torch.square(v))
    return torch.cat([w, t], dim=c_dim)


class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class DiffLoss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, yhat, y):
        loss = yhat - y
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


class AbsDiffLoss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, yhat, y):
        loss = abs(yhat) - abs(y)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss

# class GaussianLayer(nn.Module):
#     def __init__(self, k=21):
#         super(GaussianLayer, self).__init__()
#         self.conv = nn.Conv2d(3, 3, k, stride=1, padding=k // 2, bias=False, groups=3, padding_mode='replicate')
#         self.k = k
#         self.weights_init()
#
#     def forward(self, x):
#         return self.conv(x)
#
#     def weights_init(self):
#         n = np.zeros((self.k, self.k))
#         n[self.k // 2, self.k // 2] = 1
#         kk = scipy.ndimage.gaussian_filter(n, sigma=3)
#         for name, f in self.named_parameters():
#             f.data.copy_(torch.from_numpy(kk))


class StationMSE(nn.Module):
    def __init__(self, logger=None):
        super().__init__()
        self.station_mse = nn.MSELoss()

    def forward(self, pred, target, logger=None):
        mse1 = self.mean_mse(pred, target)
        if logger:
            logger.accumulate_stat(mse1)
        return mse1