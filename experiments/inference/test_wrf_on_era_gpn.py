import torch
import numpy as np
import xarray as xr

from boreylib.correction.wrf.data.my_dataloader import WRFDataset
from boreylib.correction.wrf.models.changeToERA5 import MeanToERA5
from boreylib.correction.wrf.helpers.interpolation import InvDistTree


def calc_correction_era_metric(wrf_orig, wrf_corr, era, wrf_coords, era_coords, device, interpolator_type='meaner'):
    """
    Interpolates WRF to ERA5 and calculate relative MSE reduction for WR corrected compared to WRF original
        for (u10, v10, t2) channels. Relative MSE reduction is between -inf and 1.
        1 is perfect correction, 0 means no improvement, negative values indicate deterioration of the forecast
    Parameters
    ----------
    wrf_orig: (torch.Tensor) original WRF (u10, v10, t2) tensor with shape ..., 3, h_wrf, w_wrf
    wrf_corr: (torch.Tensor) WRF (u10, v10, t2) tensor corrected by NN with shape ..., 3, h_wrf, w_wrf
    era: (torch.Tensor) ERA5 (u10, v10, t2) tensor to compare with. shape ..., 3, h_era, w_era
    wrf_coords: (torch.Tensor) flattened tensor of WRF coordinates (lon, lat) with shape h_wrf*w_wrf, 2
    era_coords: (torch.Tensor) flattened tensor of ERA5 coordinates (lon, lat) with shape h_wrf*w_wrf, 2
    device: 'cpu' or 'cuda'
    interpolator_type: 'meaner' or 'invdisttree'
        'meaner' is long to initialize, but it is what the correction model was trained with.
            It aligns each node of WRF tensor with one's nearest node of ERA5
        'invdisttree' is much faster interpolator based on scipy.KDTree. It interpolates WRF to ERA5
            based on the inverse distance to nearest N nodes. Experiments show little difference in
            metric evaluation compared to 'mean' when using this method

    Returns relative MSE reduction for each of the channel
    -------
    """
    if interpolator_type == 'invdisttree':
        interpolator = InvDistTree(x=wrf_coords, q=era_coords, device=device)
        mask = interpolator.calc_input_tensor_mask(era.shape[-2:]).flatten()
        orig_era_like = interpolator(wrf_orig.flatten(-2, -1)) * mask
        corr_era_like = interpolator(wrf_corr.flatten(-2, -1)) * mask
        era = era.flatten(-2, -1) * mask
    elif interpolator_type == 'meaner':
        interpolator = MeanToERA5(era_coords=era_coords, wrf_coords=wrf_coords, weighted=True).to(device)
        orig_era_like, corr_era_like = interpolator(wrf_orig), interpolator(wrf_corr)
        era = era.flatten(-2, -1)
        era = era[..., interpolator.mapping.unique().long()]
    else:
        raise TypeError

    criterion = torch.nn.MSELoss(reduction='none')
    channels_dim = -2
    dims_to_reduce = list(range(orig_era_like.ndim))
    dims_to_reduce.pop(channels_dim)
    orig_loss = criterion(orig_era_like, era).mean(dims_to_reduce)
    corr_loss = criterion(corr_era_like, era).mean(dims_to_reduce)

    relative_loss_reduction = (orig_loss - corr_loss) / orig_loss
    return relative_loss_reduction.numpy()


def wrf_metric_by_nc_files(wrf_orig_path, wrf_corr_path, era_path):
    wrf_orig = torch.from_numpy(np.flip(WRFDataset.load_file_vars(wrf_orig_path, ['uvmet10', 'T2']), 2).copy())
    wrf_corr = torch.from_numpy(np.flip(WRFDataset.load_file_vars(wrf_corr_path, ['U10E', 'V10E', 'T2']), 2).copy())
    era = torch.from_numpy(WRFDataset.load_file_vars(era_path, ['u10', 'v10', 't2m']))

    lonlat = WRFDataset.load_file_vars(wrf_orig_path, ['XLONG', 'XLAT'])[0]
    wrf_coords = np.flip(lonlat, 1).reshape(2, 210 * 280).T.data

    era_ds = xr.open_dataset(era_path)
    era_coords = np.stack(np.meshgrid(era_ds['longitude'].data, era_ds['latitude'].data)).reshape(2, 67 * 215).T
    res = calc_correction_era_metric(wrf_orig, wrf_corr, era,
                                     wrf_coords, era_coords,
                                     'cpu', 'invdisttree')
    return res


if __name__ == '__main__':
    wrf_metric_by_nc_files('/home/wrf_data/wrfout_d01_2022-09-08_00:00:00',
                           '/home/wrf_data/wrfout_d01_2022-09-08_00:00:00',
                           '/home/era_data/era_uv10_t2_2022-09-08')
