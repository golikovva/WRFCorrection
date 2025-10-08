import sys
import os
import random

import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

sys.path.insert(0, '../../')
from correction.helpers import plot_utils
from correction.data.data_utils import get_novaya_zemlya_mask
from correction.models.loss import RMSELoss, DiffLoss, uvt_to_wt, interp_nwp_in_time, SmallScaleLoss
from correction.helpers.interpolation import InvDistTree
from correction.helpers.metrics import NormSSIM, NamedDictMetric, normalized, channel_meaned, MeanerMetric, MulticlassAccuracy
from correction.helpers.aggregators import SpatialAggregator, AverageAggregator
from correction.helpers.ssim import CustomSSIM


def test(model, losses, dataloader, logger, cfg):
    for channel in ['u10', 'v10', 't2', 'era', 'stations', 'scatter']:
        os.makedirs(os.path.join(logger.save_dir, 'plots', channel), exist_ok=True)
    with torch.no_grad():
        model.eval()

        stat = {}
        df_stat = pd.DataFrame()
        mae = torch.nn.L1Loss(reduction='none')
        mse = torch.nn.MSELoss(reduction='none')

        metrics_dict = {
            'mesoscale_loss': NamedDictMetric(SmallScaleLoss(reduction='none', device=cfg.device), ['wrf', 'corr']),
            'era_mse': NamedDictMetric(mse, ['corr', 'era_up']),
            'era_mae': NamedDictMetric(mae, ['corr', 'era_up']),
            'orig_era_mse': NamedDictMetric(mse, ['wrf', 'era_up']),
            'orig_era_mae': NamedDictMetric(mae, ['wrf', 'era_up']),

            'mean_era_mse': NamedDictMetric(mse, ['corr_meaned', 'era']),
            'mean_era_mae': NamedDictMetric(mae, ['corr_meaned', 'era']),
            'mean_orig_era_mse': NamedDictMetric(mse,['wrf_meaned', 'era']),
            'mean_orig_era_mae': NamedDictMetric(mae, ['wrf_meaned', 'era']),

            'orig_ssim_era': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=False, channel=3).forward),
                                             ['wrf', 'era_up', 'era_up']),
            'orig_ssim_custom': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=False, channel=3, exp_coefs=(1, 1, 1)).forward),
                                               ['wrf', 'wrf', 'era_up']),
            'orig_ssim_custom_211': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=False, channel=3, exp_coefs=(2, 1, 1)).forward), 
                                                    ['wrf', 'wrf', 'era_up']),
            'ssim_custom_211': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=False, channel=3, exp_coefs=(2, 1, 1)).forward),
                                               ['corr', 'wrf', 'era_up']),
            'ssim_wrf011': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=False, channel=3,
                                                                 exp_coefs=(0, 1, 1)).forward),
                                           ['corr', 'wrf', 'wrf']),
            'ssim_wrf': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=False, channel=3, exp_coefs=(1, 1, 1)).forward),
                                        ['corr', 'wrf', 'wrf']),
            'ssim_era': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=False, channel=3, exp_coefs=(1, 1, 1)).forward),
                                        ['corr', 'era_up', 'era_up']),
            'ssim_custom': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=False, channel=3, exp_coefs=(1, 1, 1)).forward),
                                           ['corr', 'wrf', 'era_up']),

            'stations_mse': NamedDictMetric(mse, ['corr_stations_wt', 'stations_wt']),
            'stations_mae': NamedDictMetric(mae, ['corr_stations_wt', 'stations_wt']),
            'orig_stations_mse': NamedDictMetric(mse, ['wrf_stations_wt', 'stations_wt']),
            'orig_stations_mae': NamedDictMetric(mae, ['wrf_stations_wt', 'stations_wt']),
            'era_stations_mse': NamedDictMetric(mse, ['era_stations_wt', 'stations_wt']),
            'era_stations_mae': NamedDictMetric(mae, ['era_stations_wt', 'stations_wt']),
            'stations_accuracy': NamedDictMetric(MulticlassAccuracy(), ['corr_stations_dir', 'stations_dir']),
            'orig_stations_accuracy': NamedDictMetric(MulticlassAccuracy(), ['wrf_stations_dir', 'stations_dir']),
            'era_stations_accuracy': NamedDictMetric(MulticlassAccuracy(), ['era_stations_dir', 'stations_dir']),
            'scatter_mse': NamedDictMetric(mse, ['corr_scatter', 'scatter']),
            'scatter_mae': NamedDictMetric(mae, ['corr_scatter', 'scatter']),
            'orig_scatter_mse': NamedDictMetric(mse, ['wrf_scatter', 'scatter']),
            'orig_scatter_mae': NamedDictMetric(mae, ['wrf_scatter', 'scatter']),
            'era_scatter_mse': NamedDictMetric(mse, ['era_scatter', 'scatter']),
            'era_scatter_mae': NamedDictMetric(mae, ['era_scatter', 'scatter']),
            'wrf_spectrum': NamedDictMetric(torch.nn.Identity(), ['wrf_spectrum']),
            'era_spectrum': NamedDictMetric(torch.nn.Identity(), ['era_spectrum']),
            'corr_spectrum': NamedDictMetric(torch.nn.Identity(), ['corr_spectrum']),
        }

        use_scatter = hasattr(dataloader.dataset.datasets[3], 'src_grid')
        use_station = hasattr(dataloader.dataset.datasets[2], 'src_grid')
        wrf_grid, era_grid = dataloader.dataset.datasets[0].src_grid, dataloader.dataset.datasets[1].src_grid

        era_coords = np.stack([era_grid['longitude'].flatten(), era_grid['latitude'].flatten()]).T
        wrf_coords = np.stack([wrf_grid['longitude'].flatten(), wrf_grid['latitude'].flatten()]).T

        era_upsampler = InvDistTree(x=era_coords, q=wrf_coords, device=cfg.device)

        if use_scatter:
            scat_grid = dataloader.dataset.datasets[3].src_grid
            scat_coords = np.stack([scat_grid['longitude'].flatten(), scat_grid['latitude'].flatten()]).T
            scatter_interpolator = InvDistTree(x=wrf_coords, q=scat_coords, device=cfg.device)
            era_scatter_interpolator = InvDistTree(x=era_coords, q=scat_coords, device=cfg.device)
        if use_station:
            station_grid = dataloader.dataset.datasets[2].src_grid
            station_coords = np.stack([station_grid['longitude'].flatten(), station_grid['latitude'].flatten()]).T
            interpolator = InvDistTree(x=wrf_coords, q=station_coords, device=cfg.device)
            era_interpolator = InvDistTree(x=era_coords, q=station_coords, device=cfg.device)
        t = 0
        
        aggregators = [SpatialAggregator()]
        results = {metric_name: {agg.__class__.__name__: None for agg in aggregators} for metric_name in metrics_dict}

        for test_data, test_label, station, scatter, dates in (pbar := tqdm(dataloader)):
            test_data = torch.swapaxes(test_data.type(torch.float).to(cfg.device), 0, 1).contiguous()
            test_label = torch.swapaxes(test_label.type(torch.float).to(cfg.device), 0, 1)
            era_h, era_w = test_label.shape[-2:]

            if use_station:
                station = torch.permute(station.type(torch.float).to(cfg.device), (1, 0, 3, 2))

            if use_scatter:
                batch_dates = torch.as_tensor(dates.astype('datetime64[s]').astype('float64')).to(cfg.device)
                scatter_times = scatter[0].to(cfg.device).type(torch.double)
                scatter_data = torch.stack((scatter[1], scatter[2]), dim=-3).type(torch.float).to(cfg.device)

            date = dates.astype(str)
            sl = test_data.shape[0]

            dates = dates[:, None] + np.arange(sl).astype('timedelta64[h]')

            days = dates.astype('datetime64[D]')
            day_of_year = (days - days.astype('datetime64[Y]')).astype(int)
            day_of_year = np.swapaxes(day_of_year, 0, 1)

            # pbar.set_description(f'{date}_{day_of_year}')

            output = model(test_data, day_of_year).type(torch.float)
            wrf_meaned = input_to_era_map(test_data, losses.meaner, era_map_shape=(era_h, era_w))
            corr_meaned = input_to_era_map(output, losses.meaner, era_map_shape=(era_h, era_w))

            stations_wt = station[..., [0, 2], :]
            stations_dir = station[..., 1, :].unsqueeze(-2)
            # print(torch.unique(stations_dir, return_inverse=False, return_counts=False), 'station dir unique values')
            wrf_stations = input_to_stations(test_data, interpolator)
            wrf_stations_wt, wrf_stations_dir = split_uvt_to_speed_temp_and_dir(wrf_stations)
            corr_stations = input_to_stations(output, interpolator)
            corr_stations_wt, corr_stations_dir = split_uvt_to_speed_temp_and_dir(corr_stations)
            # print(stations_dir[0,0,:2], 'station dir first value')
            # print(corr_stations_dir[0,0], 'corr stations dir first value')
            # print(corr_stations[0,0,:2], 'corr stations first value')

            era_stations = input_to_stations(test_label, era_interpolator) 
            era_stations_wt, era_stations_dir = split_uvt_to_speed_temp_and_dir(era_stations)

            scatter_mask = scatter_interpolator.calc_input_tensor_mask(scatter_times.shape[-2:], 
                                                                       distance_criterion=0.15,
                                                                       fill_value=torch.nan)
            wrf_scatter = input_to_scatter(test_data, scatter_interpolator, scatter_times, batch_dates, mask=scatter_mask)
            corr_scatter = input_to_scatter(output, scatter_interpolator, scatter_times, batch_dates, mask=scatter_mask)

            era_scatter = input_to_scatter(test_label, era_scatter_interpolator, scatter_times, batch_dates, mask=scatter_mask)
            era_upsampled = era_upsampler(test_label.flatten(-2, -1)).view(test_data.shape)
            spectrum_bins, era_spectrum = get_power_spectrum(uvt_to_wt(era_upsampled, -3).squeeze().cpu())

            wrf_spectrum = get_power_spectrum(uvt_to_wt(test_data, -3).squeeze().cpu())[1]
            corr_spectrum = get_power_spectrum(uvt_to_wt(output, -3).squeeze().cpu())[1]
            era_spectrum, wrf_spectrum, corr_spectrum = map(torch.from_numpy,
                                                            [era_spectrum, wrf_spectrum, corr_spectrum])

            samples_dict = {'wrf': test_data,
                            'era_up': era_upsampled,
                            'era': test_label,
                            'corr': output,
                            'wrf_meaned': wrf_meaned,
                            'corr_meaned': corr_meaned,
                            'wrf_stations_wt': wrf_stations_wt,
                            'wrf_stations_dir': wrf_stations_dir,
                            'corr_stations_wt': corr_stations_wt,
                            'corr_stations_dir': corr_stations_dir,
                            'era_stations_wt': era_stations_wt,
                            'era_stations_dir': era_stations_dir,
                            'wrf_scatter': wrf_scatter,
                            'corr_scatter': corr_scatter,
                            'era_scatter': era_scatter,
                            'stations_wt': stations_wt,
                            'stations_dir': stations_dir,
                            'scatter': scatter_data,
                            'wrf_spectrum': wrf_spectrum,
                            'corr_spectrum': corr_spectrum,
                            'era_spectrum': era_spectrum,
                            }
            for metric_name in metrics_dict:
                err_field = metrics_dict[metric_name].calculate(samples_dict)

                for agg in aggregators:
                    agg_name = agg.__class__.__name__
                    acc = results[metric_name][agg_name]
                    if acc is None:
                        acc = agg.init_accumulator(err_field.shape[2:])
                        results[metric_name][agg_name] = acc
                    agg.accumulate(acc, err_field, dates)

            stat = append_nz_wind_statistics(stat, {'era': era_upsampled, 'wrf': test_data, 'corr': output})
            df_stat = pd.concat([df_stat, pd.DataFrame(stat)], ignore_index=True)

        l = [(metric_name, channel) for metric_name in metrics_dict for channel in ['u10', 'v10', 't2']]
        for item in l:
            if ('scatter' in item[0]) and ('t2' in item[1]):
                l.remove(item)
            if ('spectrum' in item[0]) and ('v10' in item[1]):
                l.remove(item)
            if ('station' in item[0]) and ('v10' in item[1]):
                l.remove(item)
        for item in l:        
            if ('accuracy' in item[0]) and ('t2' in item[1]):
                # print(f'Removing {item} from metrics')
                l.remove(item)

        metrics_df = pd.DataFrame(
            torch.cat([AverageAggregator.finalize(results[metric_name]['SpatialAggregator']) for metric_name in metrics_dict]).cpu().numpy()[None],
            # torch.cat([metrics_dict[metric_name].compute() for metric_name in metrics_dict]).cpu().numpy()[None],
            columns=l, index=[logger.experiment_number])
        metrics_df.columns = pd.MultiIndex.from_tuples(metrics_df.columns, names=['metric', 'channel'])
        metrics_df.to_csv(os.path.join(logger.save_dir, 'experiment_metrics'))


        era_spectrum = SpatialAggregator.finalize(results['era_spectrum']['SpatialAggregator'])
        wrf_spectrum = SpatialAggregator.finalize(results['wrf_spectrum']['SpatialAggregator'])
        corr_spectrum = SpatialAggregator.finalize(results['corr_spectrum']['SpatialAggregator'])

        for i, c in enumerate(['w10', 't2']):
            spectrum_plot = plot_utils.power_loglog_spectrum([era_spectrum[i], wrf_spectrum[i], corr_spectrum[i]],
                                                             ['era5', 'wrf', 'wrf_corr'], spectrum_bins, name=c)
            plt.savefig(os.path.join(logger.save_dir, 'plots', f'{c}_spectrum_plot'), dpi=300, bbox_inches="tight", format="pdf",)
        plt.close('all')

def angle_to_sector_class(angle: torch.Tensor, num_sectors: int = 16) -> torch.Tensor:
    """
    Convert meteorological wind direction in degrees [0,360)
    into one of `num_sectors` integer classes [0..num_sectors-1],
    each centered at multiples of 360/num_sectors.
    """
    sector_size = 360.0 / num_sectors
    # shift by half a sector so boundaries fall midway between centers
    idx = torch.floor((angle + sector_size/2) / sector_size)
    return (idx % num_sectors).long()

def split_uvt_to_speed_temp_and_dir(data: torch.Tensor, num_sectors: int = 16):
    """
    Given data of shape [..., 3, N] with channels (u, v, t):
      - compute speed = sqrt(u^2+v^2)
      - keep temperature = t
      - compute meteorological wind direction (from which it blows)
        in degrees: angle = atan2(-u, -v) → convert to [0,360)
      - map angle → discrete sector class [0..num_sectors-1]
    
    Returns:
        data_station_wt: Tensor [..., 2, N] (speed, temperature)
        dir_class:       LongTensor [..., N]  (0..num_sectors-1)
    """
    u = data[..., 0, :]
    v = data[..., 1, :]
    t = data[..., 2, :]
    
    speed = torch.sqrt(u**2 + v**2)
    temperature = t
    
    # atan2 returns radians; meteorological direction is "from" north-clockwise
    angle = (torch.atan2(-u, -v) * 180.0 / torch.pi + 360.0) % 360.0
    dir_class = angle_to_sector_class(angle, num_sectors).unsqueeze(-2)
    
    data_station_wt = torch.stack((speed, temperature), dim=-2)
    return data_station_wt, dir_class

def append_nz_wind_statistics(res_dict, models_dict, channel_dim=-3):
    nz_polygon_array = get_novaya_zemlya_mask()
    for name in models_dict:
        model_data = uvt_to_wt(models_dict[name], channel_dim)
        model_nz = nz_polygon_array * model_data.cpu()
        model_wind = model_nz.select(channel_dim, 0).numpy()
        res_dict[name + '_nz_mean'] = np.nanmean(model_wind, axis=(-2, -1)).tolist()
        res_dict[name + '_nz_mean_sq'] = np.nanmean(model_wind ** 2, axis=(-2, -1)).tolist()
        res_dict[name + '_nz_median'] = np.nanmedian(model_wind, axis=(-2, -1)).tolist()
        res_dict[name + '_nz_std'] = np.nanstd(model_wind, axis=(-2, -1)).tolist()
        res_dict[name + '_nz_percentile_2'] = np.nanpercentile(model_wind, 2, axis=(-2, -1)).tolist()
        res_dict[name + '_nz_percentile_98'] = np.nanpercentile(model_wind, 98, axis=(-2, -1)).tolist()
    # df = pd.concat([df, pd.DataFrame(res)], ignore_index=True)
    return res_dict


def calc_station_loss(wrf, stations, interpolator, loss):
    s = wrf.shape
    wrf_interpolated = interpolator(wrf.flatten(-2, -1))
    # wrf_interpolated.shape == 4, bs, 3, 46 ; stations.shape == 4, bs, 2, 46

    t2_loss = loss(wrf_interpolated[..., 2, :], stations[..., 1, :])
    wspd = torch.sqrt(torch.square(wrf_interpolated[..., 0, :]) + torch.square(wrf_interpolated[..., 1, :]))
    w10_loss = loss(wspd, stations[..., 0, :])

    return torch.stack((t2_loss, w10_loss), dim=-2)  # sl, bs, c, N_stations


def calculate_station_metric(input_orig, input_corr, stations, interpolator_orig, interpolator_corr, loss):
    orig_loss = calc_station_loss(input_orig, stations, interpolator_orig, loss)
    corr_loss = calc_station_loss(input_corr, stations, interpolator_corr, loss)

    metric = _metric(orig_loss, corr_loss)
    mean_by_time = orig_loss.mean((0, 1)), corr_loss.mean((0, 1)), metric.mean((0, 1))  # N_stations (46), 2
    mean_by_space = orig_loss.mean(-2).flatten(-2, -1), corr_loss.mean(-2).flatten(-2, -1), \
                    metric.mean(-2).flatten(-2, -1)  # bs*4, 2
    return mean_by_space, mean_by_time


def calculate_era_loss(wrf, era, meaner, criterion):
    wrf_orig = meaner(wrf)
    era = era.flatten(-2, -1)
    era = era[..., meaner.mapping.unique().long()]
    loss = criterion(wrf_orig, era)
    return loss  # loss.shape = 4, 1, 3, 8744 i.e. sl, bs, c, N


def calculate_era_metric(wrf_orig, wrf_corr, era, meaner, criterion):
    loss_orig = calculate_era_loss(wrf_orig, era, meaner, criterion)
    loss_corr = calculate_era_loss(wrf_corr, era, meaner, criterion)
    metric = _metric(loss_orig, loss_corr)
    return loss_orig, loss_corr, metric


def get_meaned_metrics(wrf_orig, wrf_corr, era, meaner, criterion):
    # loss_orig.shape = bs, 4, 3, N
    loss_orig, loss_corr, metric = calculate_era_metric(wrf_orig, wrf_corr, era, meaner, criterion)
    mean_by_time = loss_orig.mean((0, 1)), loss_corr.mean((0, 1)), metric.mean((0, 1))  # 3, N (8744)
    mean_by_space = loss_orig.mean(-1).flatten(0, 1), loss_corr.mean(-1).flatten(0, 1), \
                    metric.mean(-1).flatten(0, 1)  # bs * 4, 3
    return mean_by_space, mean_by_time


def _metric(orig, corr):
    return (orig - corr) / orig


def get_season(month):
    return month // 3 % 4


def get_season_mean_losses(orig, corr, month, sl=4):
    seasons = get_season(month)
    orig_means_by_t, corr_means_by_t = [], []
    orig_means, corr_means = [], []
    for cur_season in range(4):
        i = torch.where(seasons == cur_season)[0] * sl
        season_ids = torch.cat([i + j for j in range(sl)])

        orig_means_by_t.append(torch.nanmean(orig[season_ids], dim=0))
        corr_means_by_t.append(torch.nanmean(corr[season_ids], dim=0))
        orig_means.append(torch.nanmean(orig[season_ids], dim=[0, -1]))
        corr_means.append(torch.nanmean(corr[season_ids], dim=[0, -1]))
    losses_meaned_by_t = list(map(torch.stack, [orig_means_by_t, corr_means_by_t]))
    losses_mean = list(map(torch.stack, [orig_means, corr_means]))
    return losses_mean, losses_meaned_by_t


def get_season_mean_scatter(losses, counts, month):
    seasons = get_season(month)
    losses_means = []
    for cur_season in range(4):
        season_ids = torch.where(seasons == cur_season)[0]
        means = losses[season_ids].sum(0) / counts[season_ids].sum(0)
        means[means == torch.inf] = torch.nan
        losses_means.append(means)
    means = losses.sum(0) / counts.sum(0)
    means[means == torch.inf] = torch.nan
    losses_means.append(means)
    return torch.stack(losses_means)


def era_vector_to_map(era_vector, meaner, era_map_shape=None):
    era_map_shape = torch.Size([era_map_shape]) if era_map_shape is not None else torch.Size([67 * 215])
    base = torch.zeros([*era_vector.shape[:-1] + era_map_shape])
    base[..., meaner.mapping.unique().long().cpu()] = era_vector.float()
    return base

def input_to_era_map(data, meaner, era_map_shape=None):
    era_map_shape = torch.Size(era_map_shape) if era_map_shape is not None else torch.Size([67, 215])
    out = (meaner(data, masked=False)*torch.where(meaner.mask, 1, torch.nan)).unflatten(-1, era_map_shape)
    return out

def input_to_stations(data, interpolator):
    return interpolator(data.flatten(-2, -1))

def input_to_scatter(data, interpolator, scatter_times, data_dates, mask=None, distance_criterion=0.15):
    data_on_scat_grid = interpolator(data.flatten(-2, -1)).unflatten(dim=-1, sizes=scatter_times.shape[-2:])[:, :, :2]
    data_on_scat_grid = interp_nwp_in_time(data_on_scat_grid, scatter_times, data_dates)
    mask = interpolator.calc_input_tensor_mask(scatter_times.shape[-2:], 
                                               distance_criterion=distance_criterion,
                                               fill_value=torch.nan) if mask is None else mask
    data_on_scat_grid = data_on_scat_grid * mask
    return data_on_scat_grid

def calc_era5_error_map(wrf, era, meaner):
    t = era.flatten(-2, -1)
    t = t[..., meaner.mapping.unique().long()]
    err = torch.nn.L1Loss(reduction='none')(t, meaner(wrf))
    base = torch.zeros_like(era.flatten(-2, -1))
    base[..., meaner.mapping.unique().long()] = err.float()
    return base


def calc_scatter_error_map(data, scatter, criterion, scatter_times, data_dates, interpolator, input_mask):
    corr_on_scat_grid = interpolator(data.flatten(-2, -1)).unflatten(dim=-1, sizes=scatter.shape[-2:])[:, :, :2]
    # interpolate nwp in time
    corr_on_scat_grid = interp_nwp_in_time(corr_on_scat_grid, scatter_times, data_dates, return_counts=True)
    corr_on_scat_grid = corr_on_scat_grid #* self.wrf_mask
    # filter NaNs
    mask = (torch.isfinite(corr_on_scat_grid)) & (torch.isfinite(scatter))

    # num_valid = mask.sum().item()
    # corr_on_scat_grid = corr_on_scat_grid[mask]     # 1D tensor of only the finite entries
    # scatter = scatter[mask]
    err = criterion(corr_on_scat_grid, scatter)

    # # todo по идее правильно спрашивать criterion по которому считаем лосс, а функцию интерполяции брать самому изнутри
    # # wrf_scattered.shape == 1, 2, 2, 56760 == bs, t, c, h*w ; counts.shape == 1, 2, 56760
    # data_scattered, scatter, counts = interpolate_input_to_scat(data[..., :2, :, :], scatter, interpolator,
    #                                                             i, start_date, input_mask, return_counts=True)
    # # scatter.shape == 1, 2, 4, 132, 430 ; err.shape == 1, 2, 2, 56760
    # # err = torch.nn.L1Loss(reduction='none')(data_scattered, scatter)
    # err = criterion(data_scattered, scatter)
    # assert not torch.isnan(err).any()
    return err, mask.sum(dim=1)


def get_power_spectrum(image):
    s = image.shape
    h, w = image.shape[-2:]
    fourier_image = np.fft.fftn(image, axes=(-2, -1))
    fourier_amplitudes = np.abs(fourier_image) ** 2
    kfreqh = np.fft.fftfreq(h) * h
    kfreqw = np.fft.fftfreq(w) * w
    kfreq2D = np.meshgrid(kfreqw, kfreqh)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.reshape(np.prod(fourier_amplitudes.shape[:-2]), h * w)
    kbins = np.arange(0.5, min(h, w) // 2 + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic="mean",
                                         bins=kbins)
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    Abins = Abins.reshape(*s[:-2], -1)
    return kvals, Abins


class LossesAccumulator:
    def __init__(self, names):
        self.data = {names[i]: [] for i in range(len(names))}

    def cat_accumulate_losses(self, names, losses):
        for i, name in enumerate(names):
            if type(losses[i]) is not torch.Tensor:
                losses[i] = torch.tensor([losses[i]])
            self.data[names[i]].append(losses[i].cpu())

    def sum_accumulate_losses(self, names, losses):
        for i in range(len(names)):
            if len(self.data[names[i]]) == 0:
                self.data[names[i]] = losses[i].cpu()
            else:
                self.data[names[i]] += losses[i].cpu()

    def cat_losses(self, names):
        for name in names:
            self.data[name] = torch.cat(self.data[name])

    def save_data(self, dir_path, keys=None):
        keys = keys if keys is not None else self.data.keys()
        for name in keys:
            torch.save(self.data[name], os.path.join(dir_path, f'{name}'))


if __name__ == '__main__':
    import torch
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import numpy as np

    sys.path.insert(0, '../../')

    from correction.data.train_test_split import split_dates
    from correction.data.data_utils import WRFs2sDataset, ERAs2sDataset, ScatterDataset, StationsDataset, StackDataset, \
        dataset_with_indices, Sampler, variable_len_collate
    from correction.models.ano_corr_accumulator import AccumCorrector
    from correction.models.loss import UpdatedTurbulentMSE
    from correction.models.changeToERA5 import MeanToERA5, ClusterMapper
    from correction.helpers.interpolation import InvDistTree
    from correction.data.logger import WRFLogger
    from correction.config.cfg import cfg

    cfg['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_name = 'ano_baseline'
    logger = WRFLogger(cfg, cfg.data.logs_folder, folder_name)
    betas = [cfg.betas.beta1, cfg.betas.beta2, cfg.betas.beta3_t2, cfg.betas.beta4]
    print(betas, 'betas')

    max_sl = cfg.s2s.sequence_len
    wrf_variables = ['uvmet10', 'T2']
    era_variables = ['u10', 'v10', 't2m']
    wrf_dataset = WRFs2sDataset(cfg.data.wrf_folder, wrf_variables, seq_len=max_sl, 
                                add_coords=False,
                                add_time_encoding=False)
    era_dataset = ERAs2sDataset(cfg.data.era_folder, era_variables, seq_len=max_sl)
    st_ds = StationsDataset(cfg.data.stations_folder,seq_len=max_sl)
    sc_ds = ScatterDataset(cfg.data.scatter_folder, seq_len=max_sl)
    dataset = dataset_with_indices(StackDataset)(wrf_dataset, era_dataset, st_ds, sc_ds)

    start_date, end_date = np.datetime64(cfg.data.start_date), np.datetime64(cfg.data.end_date)
    _, _, test_days = split_dates(start_date, end_date, 0.7, 0.1, 0.2)
    print('Split completed!')



    wrf_grid, era_grid = wrf_dataset.src_grid, era_dataset.src_grid
    era_coords = np.stack([era_grid['longitude'].flatten(), era_grid['latitude'].flatten()]).T
    wrf_coords = np.stack([wrf_grid['longitude'].flatten(), wrf_grid['latitude'].flatten()]).T
    scat_grid = sc_ds.src_grid
    scat_coords = np.stack([scat_grid['longitude'].flatten(), scat_grid['latitude'].flatten()]).T
    meaner = ClusterMapper(mapping_file=None,
                           target_coords=era_coords, input_coords=wrf_coords, 
                           weighted=cfg.run_config.weighted_meaner, 
                           save_mapping=True, save_name='meaner_mapping.npy', 
                           device=cfg.device, distance_metric='euclidean').to(cfg.device)
    stations_interpolator = InvDistTree(x=wrf_coords, q=st_ds.coords, device=cfg.device) #if False else None
    scat_interpolator = InvDistTree(x=wrf_coords, q=scat_coords, device=cfg.device) #if False else None
 
    criterion = UpdatedTurbulentMSE(meaner, betas, stations_interpolator, scat_interpolator,
                                    logger=logger,kernel_type=cfg.loss_config.loss_kernel,
                                    k=cfg.loss_config.k, device=cfg.device).to(cfg.device).float()
    
    test_sampler = Sampler(test_days, shuffle=False)
    collate_fn = variable_len_collate if cfg.run_config.variable_sequence_length else variable_len_collate
    test_dataloader = DataLoader(dataset, batch_size=cfg.run_config.batch_size, num_workers=cfg.run_config.num_workers,
                                 sampler=test_sampler, collate_fn=collate_fn, pin_memory=True)

    model = AccumCorrector(os.path.join(cfg.data.logs_folder, 'ano_baseline',
                                        'day_correction_fields_conv_meaned.npy'), )

    test(model, criterion, test_dataloader, logger, cfg)
