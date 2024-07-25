import sys
import os
import torch

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, '../../')
from correction.helpers import plot_utils
from correction.models.loss import RMSELoss, DiffLoss
from correction.helpers.interpolation import InvDistTree, interpolate_input_to_scat


def test(model, losses, wrf_scaler, era_scaler, dataloader, logger, cfg):
    for channel in ['u10', 'v10', 't2', 'era', 'stations', 'scatter']:
        os.makedirs(os.path.join(logger.save_dir, 'plots', channel), exist_ok=True)
    with torch.no_grad():
        model.eval()
        losses_to_cat = ['year', 'month', 'day', 'hour', 'mesoscale_loss',  # 'wrf_orig', 'wrf_corr',
                         'wrf_orig-era', 'wrf_corr-era', 'wrf_orig-stations', 'wrf_corr-stations',
                         'wrf_orig-scatter', 'wrf_corr-scatter', 'wrf-scatter-counts',
                         'era-stations', 'era-scatter', 'era-scatter-counts',
                         ]
        acc = LossesAccumulator(names=losses_to_cat)
        dataset = dataloader.dataset
        diff = DiffLoss(reduction='none')
        mae = torch.nn.L1Loss(reduction='none')
        rmse = RMSELoss(reduction='none')

        metadata = dataset.metadata
        era_coords = np.stack([metadata['era_xx'].flatten(), metadata['era_yy'].flatten()]).T
        scat_coords = np.stack([metadata['scat_xx'].flatten(), metadata['scat_yy'].flatten()]).T
        wrf_coords = np.stack([metadata['wrf_xx'].flatten(), metadata['wrf_yy'].flatten()]).T
        interpolator = InvDistTree(x=wrf_coords, q=metadata['coords'])
        era_interpolator = InvDistTree(x=era_coords, q=metadata['coords'])
        era_scatter_interpolator = InvDistTree(x=era_coords, q=scat_coords)
        t = 0
        for test_data, test_label, station, scatter, date_id in tqdm(dataloader, total=len(dataloader) // 4):
            test_data = torch.swapaxes(test_data.type(torch.float).to(cfg.device), 0, 1)
            test_label = torch.swapaxes(test_label.type(torch.float).to(cfg.device), 0, 1)
            station = torch.permute(station.type(torch.float).to(cfg.device), (1, 0, 3, 2))[..., [3, 1], :]
            scatter = scatter.to(cfg.device)

            datefile_id, hour = dataset.get_path_id(date_id)
            datefile_id, hour = datefile_id.item(), hour.item()
            date = dataset.wrf_files[datefile_id].split('_')[-2]
            year, month, day = map(int, date.split('-'))
            # dates.append(date)

            test_data = wrf_scaler.transform(test_data, dims=2)

            if 'lfreq' in cfg.model_type:
                _, l_freq_corr, h_freq = model(test_data)
                output = l_freq_corr + h_freq
            else:
                output = model(test_data)

            output = era_scaler.inverse_transform(output, dims=2)
            test_data = wrf_scaler.inverse_transform(test_data, dims=2)[:, :, :3]

            mesoscale_loss = losses(output, test_data, expanded_out=True)[2].item()

            # конкатенируем и сохраняем все ошибки, потом их интерпретируем. требует много места на диске тк
            # сохраняются массивы соразмерные размеру датасета
            # orig ошибку нужно сохранить лишь раз а затем подгружать и пользоваться

            # wrf era difference
            orig_era = calculate_era_loss(test_data, test_label, losses.meaner, rmse).flatten(0, 1)
            corr_era = calculate_era_loss(output, test_label, losses.meaner, rmse).flatten(0, 1)
            # wrf stations difference
            orig_stations = calc_station_loss(test_data, station, interpolator, rmse).flatten(0, 1)
            corr_stations = calc_station_loss(output, station, interpolator, rmse).flatten(0, 1)

            # wrf scatter difference
            orig_scatter, orig_counts = calc_scatter_error_map(test_data, scatter, rmse, date_id,
                                                               metadata['start_date'],
                                                               losses.scatter_interpolator, losses.wrf_mask)
            corr_scatter, corr_counts = calc_scatter_error_map(output, scatter, rmse, date_id, metadata['start_date'],
                                                               losses.scatter_interpolator, losses.wrf_mask)
            # era stations difference
            era_stations = calc_station_loss(test_label, station, era_interpolator, rmse).flatten(0, 1)
            # era scatter difference
            era_scatter, era_counts = calc_scatter_error_map(test_label, scatter, rmse, date_id, metadata['start_date'],
                                                             era_scatter_interpolator, losses.wrf_mask)
            acc.cat_accumulate_losses(names=losses_to_cat, losses=[year, month, day, hour, mesoscale_loss,
                                                                   # test_data, output,
                                                                   orig_era, corr_era,
                                                                   orig_stations, corr_stations,
                                                                   orig_scatter.sum(1), corr_scatter.sum(1),
                                                                   orig_counts,
                                                                   era_stations, era_scatter.sum(1), era_counts])

            if cfg.test_config.draw_plots:
                if date_id % 756 == 0:
                    station_metric = _metric(orig_stations.mean([0, -1]), corr_stations.mean([0, -1]))
                    era_metric = _metric(orig_era.mean((0, -1)), corr_era.mean((0, -1)))
                    for i, channel in enumerate(['u10', 'v10', 't2']):
                        st_m = station_metric[0].item() if channel == 't2' else station_metric[1].item()
                        simple_plot = plot_utils.draw_simple_plots(test_data, output, test_label, i,
                                                                   orig_era.mean().item(),
                                                                   corr_era.mean().item(),
                                                                   era_metric[i].item(), st_m,
                                                                   f'{date} {hour}:00')
                        plt.savefig(os.path.join(logger.save_dir, 'plots', channel, f'plot_{date}_{hour}'))
                if date_id == len(dataset) - 1:
                    station_metric = _metric(orig_stations.mean([0, -1]), corr_stations.mean([0, -1]))
                    for i, channel in enumerate(['u10', 'v10', 't2']):
                        st_m = station_metric[0].item() if channel == 't2' else station_metric[1].item()
                        era_metric = _metric(orig_era.mean((0, -1)), corr_era.mean((0, -1)))
                        mega_plot = plot_utils.draw_mega_plot(test_data, test_label, output, i, date, hour,
                                                              era_metric[i].item(), st_m)
                        plt.savefig(os.path.join(logger.save_dir, 'plots', f'megaplot_{channel}'))
                plt.close('all')
        acc.cat_losses(losses_to_cat)

        print("Drawing wrf era losses hist...")
        orig_era = acc.data['wrf_orig-era']
        corr_era = acc.data['wrf_corr-era']
        losses_plot = plot_utils.draw_losses_gist(orig_era.transpose(0, 1).mean(-1),
                                                  corr_era.transpose(0, 1).mean(-1), 'ERA5')
        plt.savefig(os.path.join(logger.save_dir, 'plots', 'era', f'era_losses_hist'))
        plt.close('all')

        print("Drawing seasonal wrf era bar plot (насколько улучшились данные wrf относительно era5)...")
        wrf_era_mean_loss, wrf_era_t_mean_map = get_season_mean_losses(orig_era, corr_era, acc.data['month'])
        season_metric_bar = plot_utils.draw_seasonal_bar_plot(_metric(*wrf_era_mean_loss), dtype="WRF on ERA5")
        plt.savefig(os.path.join(logger.save_dir, 'plots', 'era', f'season_era_metric_bar_plot'))
        plt.close('all')

        # карта ошибок wrf на era5
        print('Drawing error map between wrf and era5...')
        orig_era_figs = plot_utils.draw_seasonal_era_error_map(era_vector_to_map(wrf_era_t_mean_map[0], losses.meaner),
                                                               dtype='WRF orig')
        [f.savefig(os.path.join(logger.save_dir, 'plots', 'era', f'{name}.png')) for name, f in orig_era_figs.items()]
        corr_era_figs = plot_utils.draw_seasonal_era_error_map(era_vector_to_map(wrf_era_t_mean_map[1], losses.meaner),
                                                               dtype='WRF corr')
        [f.savefig(os.path.join(logger.save_dir, 'plots', 'era', f'{name}.png')) for name, f in corr_era_figs.items()]
        plt.close('all')

        # гистограмма ошибок wrf на станциях
        print('Drawing wrf error hist on stations...')
        # print(orig_stations.shape)
        orig_stations = acc.data['wrf_orig-stations']
        corr_stations = acc.data['wrf_corr-stations']
        losses_plot = plot_utils.draw_losses_gist(orig_stations.transpose(0, 1).mean(-1),
                                                  corr_stations.transpose(0, 1).mean(-1),
                                                  channels=['t2', 'w10'], dtype='Stations')
        plt.savefig(os.path.join(logger.save_dir, 'plots', 'stations', f'station_losses_hist'))
        plt.close('all')

        # усредненная метрика wrf на станциях по сезонам
        print('Drawing mean seasonal wrf station metric...')
        wrf_st_mean_loss, wrf_st_t_mean_map = get_season_mean_losses(orig_stations, corr_stations, acc.data['month'])
        season_metric_bar = plot_utils.draw_seasonal_bar_plot(_metric(*wrf_st_mean_loss), channels=['t2', 'w10'],
                                                              dtype="WRF on Stations")
        plt.savefig(os.path.join(logger.save_dir, 'plots', 'stations', f'season_wrf_stations_metric_bar_plot'))
        plt.close('all')

        # карта метрик wrf на станциях по сезонам
        print("Drawing wrf seasonal metric map on stations...")
        st_figs = plot_utils.draw_seasonal_stations_error_map(_metric(*wrf_st_t_mean_map), metadata, output, test_label)
        [f.savefig(os.path.join(logger.save_dir, 'plots', 'stations', f'{name}.png')) for name, f in st_figs.items()]
        plt.close('all')

        # усредненная метрика era5 на станциях по сезонам
        print('Drawing mean seasonal era5 station metric')
        era_stations = acc.data['era-stations']
        era_st_mean_loss, era_st_t_mean_map = get_season_mean_losses(orig_stations, era_stations, acc.data['month'])
        season_metric_bar = plot_utils.draw_seasonal_bar_plot(_metric(*era_st_mean_loss),
                                                              channels=['t2', 'w10'], dtype="ERA5 on Stations")
        plt.savefig(os.path.join(logger.save_dir, 'plots', 'stations', f'season_stations_metric_bar_plot'))
        plt.close('all')

        # карта метрик era5 на станциях по сезонам
        print('Drawing era5 seasonal metric map on stations...')
        era_st = plot_utils.draw_seasonal_stations_error_map(_metric(*era_st_t_mean_map), metadata, output, test_label,
                                                             dtype='ERA5')
        [f.savefig(os.path.join(logger.save_dir, 'plots', 'stations', f'{name}.png')) for name, f in era_st.items()]
        plt.close('all')

        # карта ошибок wrf на скаттерометре
        print('Drawing wrf seasonal error map on scatter...')
        print(acc.data['wrf_orig-scatter'].shape, acc.data['month'].shape, acc.data['wrf-scatter-counts'].shape)
        torch.save(acc.data['wrf_orig-scatter'], os.path.join(logger.save_dir, 'wrf_orig-scatter'))
        torch.save(acc.data['month'], os.path.join(logger.save_dir, 'month'))
        torch.save(acc.data['wrf-scatter-counts'], os.path.join(logger.save_dir, 'wrf-scatter-counts'))
        orig_scatter_seasonal = get_season_mean_scatter(acc.data['wrf_orig-scatter'], acc.data['wrf-scatter-counts'],
                                                        acc.data['month'])
        print(orig_scatter_seasonal.shape, 'orig scatter seasonal')
        torch.save(orig_scatter_seasonal, os.path.join(logger.save_dir, 'orig_scatter_seasonal'))
        corr_scatter_seasonal = get_season_mean_scatter(acc.data['wrf_corr-scatter'], acc.data['wrf-scatter-counts'],
                                                        acc.data['month'])
        scat_figs = plot_utils.draw_seasonal_scat_err_map(orig_scatter_seasonal, lons=metadata['scat_xx'],
                                                          lats=metadata['scat_yy'], dtype='WRF orig', colormap='common')
        [f.savefig(os.path.join(logger.save_dir, 'plots', 'scatter', f'{name}.png')) for name, f in scat_figs.items()]
        scat_figs = plot_utils.draw_seasonal_scat_err_map(corr_scatter_seasonal, lons=metadata['scat_xx'],
                                                          lats=metadata['scat_yy'], dtype='WRF corr', colormap='common')
        [f.savefig(os.path.join(logger.save_dir, 'plots', 'scatter', f'{name}.png')) for name, f in scat_figs.items()]
        plt.close('all')

        # карта ошибок era5 на скаттерометре
        print('Drawing era5 seasonal metric map on scatter...')
        era_scatter_seasonal = get_season_mean_scatter(acc.data['era-scatter'], acc.data['era-scatter-counts'],
                                                       acc.data['month'])
        scat_figs = plot_utils.draw_seasonal_scat_err_map(era_scatter_seasonal, lons=metadata['scat_xx'],
                                                          lats=metadata['scat_yy'], dtype='ERA5', colormap='common')
        [f.savefig(os.path.join(logger.save_dir, 'plots', 'scatter', f'{name}.png')) for name, f in scat_figs.items()]
        plt.close('all')

        if cfg.test_config.save_losses:
            acc.save_data(logger.save_dir)
        test_loss = wrf_era_mean_loss[1].mean(0).tolist() + wrf_st_mean_loss[1].mean(0).tolist() + \
                    corr_scatter_seasonal[-1].nanmean(-1).tolist() + [acc.data['mesoscale_loss'].mean().item()]
        test_orig_loss = wrf_era_mean_loss[0].mean(0).tolist() + wrf_st_mean_loss[0].mean(0).tolist() + \
                        orig_scatter_seasonal[-1].nanmean(-1).tolist() + [0]
        a = True
        if a:
            df = pd.DataFrame([test_loss], columns=['era_u10', 'era_v10', 'era_t2', 'st_t2', 'st_w10', 'sc_u10',
                                                    'sc_v10', 'mesoscale_loss'])
            df.to_csv(os.path.join(logger.save_dir, 'mean_losses'))
            df = pd.DataFrame([test_orig_loss], columns=['era_u10', 'era_v10', 'era_t2', 'st_t2', 'st_w10', 'sc_u10',
                                                         'sc_v10', 'mesoscale_loss'])
            df.to_csv(os.path.join(logger.save_dir, 'mean_orig_losses'))
    return test_loss


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

        orig_means_by_t.append(orig[season_ids].mean(0))
        corr_means_by_t.append(corr[season_ids].mean(0))
        orig_means.append(orig[season_ids].mean([0, -1]))
        corr_means.append(corr[season_ids].mean([0, -1]))
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
    era_map_shape = era_map_shape if era_map_shape is not None else torch.Size([67 * 215])
    base = torch.zeros(era_vector.shape[:-1] + era_map_shape)
    base[..., meaner.mapping.unique().long()] = era_vector.float()
    return base


def calc_era5_error_map(wrf, era, meaner):
    t = era.flatten(-2, -1)
    t = t[..., meaner.mapping.unique().long()]
    err = torch.nn.L1Loss(reduction='none')(t, meaner(wrf))
    base = torch.zeros_like(era.flatten(-2, -1))
    base[..., meaner.mapping.unique().long()] = err.float()
    return base


def calc_scatter_error_map(data, scatter, criterion, i, start_date, interpolator, input_mask):
    # todo по идее правильно спрашивать criterion по которому считаем лосс, а функцию интерполяции брать самому изнутри
    # wrf_scattered.shape == 1, 2, 2, 56760 == bs, t, c, h*w ; counts.shape == 1, 2, 56760
    data_scattered, scatter, counts = interpolate_input_to_scat(data[..., :2, :, :], scatter, interpolator,
                                                                i, start_date, input_mask, return_counts=True)
    # scatter.shape == 1, 2, 4, 132, 430 ; err.shape == 1, 2, 2, 56760
    # err = torch.nn.L1Loss(reduction='none')(data_scattered, scatter)
    err = criterion(data_scattered, scatter)
    assert not torch.isnan(err).any()
    return err, counts


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

    def save_data(self, dir_path):
        for name in self.data.keys():
            torch.save(self.data[name], os.path.join(dir_path, f'{name}'))
