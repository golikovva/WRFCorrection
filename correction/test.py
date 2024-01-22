import sys

sys.path.insert(0, '../')
import torch
import os
from correction.config.config import cfg
from correction.helpers import plot_utils
from correction.helpers.plot_utils import draw_simple_plots, draw_mega_plot, draw_station_metrics
from correction.helpers.interpolation import get_nearest_neighbour, Interpolator
from correction.data.train_test_split import find_files
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
import pickle
import pandas as pd


# def test_model(test_dataloader, model, wrf_scaler, era_scaler, criterion, logger):
#     test_loss = test(test_dataloader, model, criterion, wrf_scaler, era_scaler, logger)
#     print('test_loss', test_loss)


def test(model, losses, wrf_scaler, era_scaler, dataloader, logs_dir, logger=None):
    for channel in ['u10', 'v10', 't2', 'stations']:
        os.makedirs(os.path.join(logger.save_dir, 'plots', channel), exist_ok=True)
    with torch.no_grad():
        model.eval()
        print(len(dataloader))
        losses_to_accumulate = ['wrf_orig_loss', 'wrf_corr_loss', 'era_metric',
                                't2_orig_loss', 'w10_orig_loss', 't2_corr_loss', 'w10_corr_loss',
                                't2_station_metric', 'w10_station_metric',
                                'era_t2_station_metric', 'era_w10_station_metric']
        accumulator = LossesAccumulator(names=losses_to_accumulate)
        t2_stations_metrics, w10_stations_metrics = [], []
        input_losses, output_losses, metrics, metrics_0, dates = [], [], [], [], []
        dataset = dataloader.dataset

        wrf_grid_coords = torch.from_numpy(np.stack([dataset.metadata['wrf_xx'].flatten(),
                                                     dataset.metadata['wrf_yy'].flatten()]))
        era_grid_coords = torch.from_numpy(np.stack([dataset.metadata['era_xx'].flatten(),
                                                     dataset.metadata['era_yy'].flatten()]))
        stations_coords = torch.from_numpy(dataset.metadata['coords']).t()
        print(stations_coords.shape, wrf_grid_coords.shape, 'grids shapes')
        print(stations_coords.dtype, wrf_grid_coords.dtype, 'grids dtypes')
        interpolator = Interpolator(stations_coords.float(), wrf_grid_coords.float()).float()
        interpolator_era = Interpolator(stations_coords.float(), era_grid_coords.float()).float()

        for date_id in list(range(0, len(dataset), 4)):
            # for test_data, test_label, station, date_id in dataloader:
            test_data, test_label, station, _ = dataset[date_id]
            test_data, test_label = torch.from_numpy(np.expand_dims(test_data, 0)), torch.from_numpy(
                np.expand_dims(test_label, 0))
            station = torch.from_numpy(np.expand_dims(station, 1)).to(cfg.GLOBAL.DEVICE)
            datefile_id, hour = dataset.get_path_id(date_id)
            date = dataset.wrf_files[datefile_id].split('_')[-2]
            dates.append(date)
            test_data = torch.swapaxes(test_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            test_label = torch.swapaxes(test_label.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            test_data = wrf_scaler.channel_transform(test_data, 2)
            test_label = era_scaler.channel_transform(test_label, 2)

            output = model(test_data)

            _, _, metric_0 = calculate_metric(test_data[:, :, :3], output, test_label, losses)

            output = wrf_scaler.channel_inverse_transform(output, 2)
            test_data = wrf_scaler.channel_inverse_transform(test_data, 2)[:, :, :3]
            test_label = era_scaler.channel_inverse_transform(test_label, 2)

            t2_orig_loss, w10_orig_loss = calc_station_loss(test_data, station, interpolator)
            t2_corr_loss, w10_corr_loss = calc_station_loss(output, station, interpolator)

            input_loss, output_loss, metric = calculate_metric(test_data, output, test_label, losses)

            t2_station_metric, w10_station_metric = calculate_station_metric(test_data, output, station,
                                                                             interpolator, interpolator)
            erat2_station_metric, eraw10_station_metric = calculate_station_metric(test_data, test_label, station,
                                                                                   interpolator, interpolator_era)

            print('input_loss, output_loss, metric, metric_0')
            print(input_loss.item(), output_loss.item(), metric.item(), metric_0.item())
            print('erat2_station_metric, eraw10_station_metric')
            print(erat2_station_metric.item(), eraw10_station_metric.item())
            print('t2_station_metric, w10_station_metric')
            print(t2_station_metric.item(), w10_station_metric.item())

            t2_stations_metrics.append(t2_station_metric.mean().item())
            w10_stations_metrics.append(w10_station_metric.mean().item())
            metrics.append(metric.item())
            metrics_0.append(metric_0.item())
            input_losses.append(input_loss.item())
            output_losses.append(output_loss.item())

            if cfg.run_config.draw_plots:
                if date_id % 252 == 0:
                    draw_station_metrics(dataset.metadata, test_label, output, t2_station_metric, w10_station_metric)
                    plt.savefig(os.path.join(logger.save_dir, 'plots', 'stations', f'plot_{date}_{hour}'))
                    for i, channel in enumerate(['u10', 'v10', 't2']):
                        station_metric = t2_station_metric.mean().item() if channel == 't2' else w10_station_metric.mean().item()
                        simple_plot = plot_utils.draw_simple_plots(test_data, output, test_label, i,
                                                                   input_loss.mean().item(), output_loss.mean().item(), metric.mean().item(), station_metric,
                                                                   f'{date} {hour}:00')
                        plt.savefig(os.path.join(logger.save_dir, 'plots', channel, f'plot_{date}_{hour}'))
                if date_id == len(dataset) - 1:
                    for i, channel in enumerate(['u10', 'v10', 't2']):
                        station_metric = t2_station_metric.mean().item() if channel == 't2' else w10_station_metric.mean().item()
                        mega_plot = plot_utils.draw_mega_plot(test_data, test_label, output, i, date, hour,
                                                              metric.mean().item(), station_metric)
                        plt.savefig(os.path.join(logger.save_dir, 'plots', f'megaplot_{channel}'))
                    plt.close('all')

        losses_df = pd.DataFrame({'wrf_dates': dates, 'input_losses': input_losses, 'output_losses': output_losses,
                                  'metrics': metrics, 't2_stations_metrics': t2_stations_metrics,
                                  'w10_stations_metrics': w10_stations_metrics})
        losses_df.to_csv(os.path.join(logger.save_dir, 'losses_per_samples.csv'), index=False)
        metric_hist = plot_utils.make_metric_gist(losses_df)
        plt.savefig(os.path.join(logger.save_dir, 'plots', f'metric_hist'))
        season_loss_plot = plot_utils.draw_season_metric_plot(losses_df)
        plt.savefig(os.path.join(logger.save_dir, 'plots', f'season_loss_plot'))
        plt.close('all')

        t2_stations_metrics = np.array(t2_stations_metrics)
        w10_stations_metrics = np.array(w10_stations_metrics)
        metrics = np.array(metrics)
        metrics_0 = np.array(metrics_0)
        input_losses = np.array(input_losses)
        output_losses = np.array(output_losses)

        # accumulator.save_data(logger.save_dir)
        print(t2_stations_metrics.shape, 't2_stations_metrics.shape')
        pd.DataFrame({'input_losses': [input_losses.mean()], 'output_losses': [output_losses.mean()],
                      'metrics': [metrics.mean()], 't2_stations_metrics': [t2_stations_metrics.mean()],
                      'w10_stations_metrics': [w10_stations_metrics.mean()]}) \
            .to_csv(os.path.join(logger.save_dir, 'losses.csv'), index=False)

    return input_losses.mean(), output_losses.mean(), metrics.mean(), t2_stations_metrics.mean(), w10_stations_metrics.mean(), metrics_0.mean()


class LossesAccumulator:
    def __init__(self, names):
        self.data = {names[i]: [] for i in range(len(names))}

    def accumulate_losses(self, names, losses):
        for i in range(len(names)):
            print(losses[i].shape)
            self.data[names[i]].append(losses[i].cpu().numpy())

    def save_data(self, dir_path):
        for name in self.data.keys():
            np.save(os.path.join(dir_path, f'{name}.npy'), np.concatenate(self.data[name]))


def calc_station_loss(wrf, stations, interpolator):
    s = wrf.shape
    wrf_interpolated = interpolator(wrf.view(*s[:-2], s[-1] * s[-2]))
    loss = torch.nn.L1Loss(reduction='none')
    t2_loss = loss(wrf_interpolated[..., 2, :, 0], stations[..., :, 1])
    wspd = torch.sqrt(torch.square(wrf_interpolated[..., 0, :, 0]) + torch.square(wrf_interpolated[..., 1, :, 0]))
    w10_loss = loss(wspd, stations[..., :, 3])
    return t2_loss, w10_loss


def calculate_station_metric(input_orig, input_corr, stations, interpolator_orig, interpolator_corr):
    o_t2_loss, o_w10_loss = calc_station_loss(input_orig, stations, interpolator_orig)
    c_t2_loss, c_w10_loss = calc_station_loss(input_corr, stations, interpolator_corr)
    t2_metric = (o_t2_loss.mean() - c_t2_loss.mean()) / o_t2_loss.mean()
    wspd_metric = (o_w10_loss.mean() - c_w10_loss.mean()) / o_w10_loss.mean()
    return t2_metric, wspd_metric


def calculate_metric(wrf_orig, wrf_corr, era, criterion):
    loss_orig = criterion(wrf_orig, wrf_orig, era)
    loss_corr = criterion(wrf_orig, wrf_corr, era)
    metric = (loss_orig - loss_corr) / loss_orig
    return loss_orig, loss_corr, metric


def calculate_station_metric_old(wrf_orig, wrf_corr, stations, interpolator):
    s = wrf_orig.shape
    orig = interpolator(wrf_orig.view(*s[:-2], s[-1] * s[-2]))
    corr = interpolator(wrf_corr.view(*s[:-2], s[-1] * s[-2]))
    loss = torch.nn.L1Loss(reduction='mean')
    o_loss = loss(orig[..., 2, :, 0], stations[..., :, 1])
    c_loss = loss(corr[..., 2, :, 0], stations[..., :, 1])
    t2_metric = (o_loss - c_loss) / o_loss
    wspd_orig = torch.sqrt(torch.square(orig[..., 0, :, 0]) + torch.square(orig[..., 1, :, 0]))
    wspd_corr = torch.sqrt(torch.square(corr[..., 0, :, 0]) + torch.square(corr[..., 1, :, 0]))
    o_loss = loss(wspd_orig, stations[..., :, 3])
    c_loss = loss(wspd_corr, stations[..., :, 3])
    wspd_metric = (o_loss - c_loss) / o_loss
    return t2_metric, wspd_metric
