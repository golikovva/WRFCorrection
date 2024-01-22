from torch import nn
import torch
import numpy as np
import scipy


class TurbulentMSE(nn.Module):
    def __init__(self, meaner, interpolator, beta=1, beta2=1, logger=None, kernel_type='gauss', k=9):
        super().__init__()
        self.mean_mse = nn.MSELoss()
        self.delta_mse = nn.MSELoss()
        self.station_mse = nn.MSELoss()
        self.delta_conv = DeltaConvLayer(k, kernel_type=kernel_type)
        self.station_interpolator = interpolator
        self.meaner = meaner
        self.beta = beta
        self.beta2 = beta2
        if logger:
            logger.set_beta(beta)

    def forward(self, orig, corr, target, stations=None, orig_scaler=None, logger=None):
        mean_corr = self.meaner(corr)

        t = target.flatten(-2, -1)  # .view(*target.shape[:-2], target.shape[-1] * target.shape[-2])
        t = t[..., self.meaner.mapping.unique().long()]

        mse1 = self.mean_mse(mean_corr, t)
        # delta_corr = corr.permute(-2, -1, *list(range(len(corr.shape) - 2))) - corr.mean(dim=(-1, -2))
        # delta_orig = orig.permute(-2, -1, *list(range(len(orig.shape) - 2))) - orig.mean(dim=(-1, -2))
        delta_corr = corr - self.delta_conv(corr.view(-1, 3, 210, 280)).view(corr.shape)
        delta_orig = orig - self.delta_conv(orig.view(-1, 3, 210, 280)).view(orig.shape)
        mse2 = self.delta_mse(delta_corr, delta_orig)

        mse3 = 0
        if stations is not None:
            assert orig_scaler is not None, f'WRF scaler needed to calculate station loss'
            pred_stations = self.station_interpolator(orig_scaler.channel_inverse_transform(corr, 2).flatten(-2, -1))
            print(pred_stations.shape, stations.shape)
            mse3 = self.station_mse(uvt_to_wt(pred_stations), stations)

        if logger:
            logger.accumulate_stat(mse1 + self.beta * mse2 + self.beta2 * mse3, mse1, mse2, mse3)
        return mse1 + self.beta * mse2 + self.beta2 * mse3


def uvt_to_wt(pred_st, c_dim=-2):
    assert pred_st.shape[c_dim] == 3, f'assumed 3 channels (u, v, t) to be proccessed but got {pred_st.shape[c_dim]}'
    u, v, t = torch.split(pred_st, 1, dim=c_dim)
    w = torch.sqrt(torch.square(u) + torch.square(v))
    return torch.cat([w, t], dim=c_dim)


class DeltaConvLayer(nn.Module):
    def __init__(self, k=21, kernel_type='gauss'):
        super(DeltaConvLayer, self).__init__()
        self.conv = nn.Conv2d(3, 3, k, stride=1, padding=k // 2, bias=False, groups=3, padding_mode='replicate')
        self.k = k
        self.kernel_type = kernel_type
        self.weights_init(k)

    def forward(self, x):
        return self.conv(x)

    def weights_init(self, k):
        if self.kernel_type == 'mean':
            w = torch.ones_like(self.delta_conv.weight) / self.k / self.k
            self.conv.weight = nn.Parameter(w)
        elif self.kernel_type == 'gauss':
            n = np.zeros((self.k, self.k))
            n[self.k // 2, self.k // 2] = 1
            kk = scipy.ndimage.gaussian_filter(n, sigma=3)
            for name, f in self.named_parameters():
                f.data.copy_(torch.from_numpy(kk))


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
