import sys

sys.path.insert(0, '../')
import torch
import os
from correction.config.config import cfg
from correction.helpers import plot_utils
from correction.helpers.plot_utils import draw_simple_plots, draw_mega_plot
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
    with torch.no_grad():
        model.eval()
        print(len(dataloader))
        t2_stations_metrics, w10_stations_metrics = [], []
        input_losses, output_losses, metrics, metrics_0, dates = [], [], [], [], []
        dataset = dataloader.dataset

        wrf_grid_coords = torch.from_numpy(np.stack([dataset.metadata['wrf_xx'].flatten(),
                                                     dataset.metadata['wrf_xx'].flatten()]))
        stations_coords = torch.from_numpy(dataset.metadata['Coords']).t()
        print(stations_coords.shape, wrf_grid_coords.shape, 'grids shapes')
        print(stations_coords.dtype, wrf_grid_coords.dtype, 'grids dtypes')
        interpolator = Interpolator(wrf_grid_coords.float(), stations_coords.float()).float()

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

            input_loss, output_loss, metric = calculate_metric(test_data, output, test_label, losses)
            t2_station_metric, w10_station_metric = calculate_station_metric(test_data, output, station, interpolator)
            print('input_loss, output_loss, metric, metric_0, t2_station_metric, w10_stations_metric')
            print(input_loss, output_loss, metric, metric_0, t2_station_metric, w10_station_metric)

            t2_stations_metrics.append(t2_station_metric)
            w10_stations_metrics.append(w10_station_metric)
            metrics.append(metric)
            metrics_0.append(metric_0)
            input_losses.append(input_loss)
            output_losses.append(output_loss)

            if cfg.run_config.draw_plots:
                if date_id % 252 == 0:
                    for i, channel in enumerate(['u10', 'v10', 't2']):
                        station_metric = t2_station_metric if channel == 't2' else w10_station_metric
                        simple_plot = plot_utils.draw_simple_plots(test_data, output, test_label, i,
                                                                   input_loss, output_loss, metric, station_metric,
                                                                   f'{date} {hour}:00')
                        plt.savefig(os.path.join(logger.save_dir, 'plots', channel, f'plot_{date}_{hour}'))
                if date_id == len(dataset) - 1:
                    for i, channel in enumerate(['u10', 'v10', 't2']):
                        station_metric = t2_station_metric if channel == 't2' else w10_station_metric
                        mega_plot = plot_utils.draw_mega_plot(test_data, test_label, output, i, date, hour,
                                                              metric, station_metric)
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

        print(t2_stations_metrics.shape, 't2_stations_metrics.shape')
        pd.DataFrame({'input_losses': [input_losses.mean()], 'output_losses': [output_losses.mean()],
                      'metrics': [metrics.mean()], 't2_stations_metrics': [t2_stations_metrics.mean()],
                      'w10_stations_metrics': [w10_stations_metrics.mean()]}) \
            .to_csv(os.path.join(logger.save_dir, 'losses.csv'), index=False)

    return input_losses.mean(), output_losses.mean(), metrics.mean(), t2_stations_metrics.mean(), w10_stations_metrics.mean(), metrics_0.mean()


def calculate_metric(wrf_orig, wrf_corr, era, criterion):
    loss_orig = criterion(wrf_orig, wrf_orig, era)
    loss_corr = criterion(wrf_orig, wrf_corr, era)
    metric = ((loss_orig - loss_corr) / loss_orig).mean()
    return loss_orig.item(), loss_corr.item(), metric.item()


def calculate_station_metric(wrf_orig, wrf_corr, stations, interpolator):
    s = wrf_orig.shape
    orig = interpolator(wrf_orig.view(*s[:-2], s[-1] * s[-2]))
    corr = interpolator(wrf_corr.view(*s[:-2], s[-1] * s[-2]))
    print(orig.shape, corr.shape, stations.shape, 'before loss shapesss')
    o_loss = torch.nn.MSELoss()(orig[..., 2, :], stations[..., :, 1])
    c_loss = torch.nn.MSELoss()(corr[..., 2, :], stations[..., :, 1])
    t2_metric = ((o_loss - c_loss) / o_loss).mean()
    wspd_orig = torch.sqrt(torch.square(orig[..., 0, :]) + torch.square(orig[..., 0, :]))
    wspd_corr = torch.sqrt(torch.square(corr[..., 0, :]) + torch.square(corr[..., 0, :]))
    o_loss = torch.nn.MSELoss()(wspd_orig, stations[..., :, 3])
    c_loss = torch.nn.MSELoss()(wspd_corr, stations[..., :, 3])
    wspd_metric = ((o_loss - c_loss) / o_loss).mean()
    return t2_metric.item(), wspd_metric.item()


def test_model(model, losses, wrf_scaler, era_scaler, dataloader, logs_dir, logger=None, draw_plots=False):
    loss_values = np.zeros([2, len(losses)])
    # input_loss_values = 0
    i = 0
    with torch.no_grad():
        model.eval()
        for test_data, test_label in tqdm(dataloader):
            test_data = torch.swapaxes(test_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            test_label = torch.swapaxes(test_label.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            test_data = wrf_scaler.channel_transform(test_data, 2)
            test_label = era_scaler.channel_transform(test_label, 2)
            output = model(test_data)
            if cfg.run_config.use_spatiotemporal_encoding:
                test_data = test_data[:, :, :3]
            for j, loss in enumerate(losses):
                test_loss = loss(test_data, output, test_label, logger)
                input_loss = loss(test_data, test_data, test_label, logger)
                loss_values[0, j] += test_loss.item()
                loss_values[1, j] += input_loss.item()
            # input_loss_values += input_loss.item()
            if draw_plots:
                draw_simple_plots(test_data, output, test_label, input_loss, test_loss, logs_dir, i)
                i += 1
        loss_values = loss_values / len(dataloader)
        # input_loss_values = input_loss_values / len(dataloader)
        if logger:
            logger.print_stat_readable()
        np.save(os.path.join(cfg.GLOBAL.BASE_DIR, logs_dir, 'loss_values'), loss_values)
    return loss_values


def draw_advanced_plots(model, losses, wrf_scaler, era_scaler, dataset, logs_dir, logger=None):
    with torch.no_grad():
        model.eval()
        print(model)
        test_loss = 0.0
        test_losses = []
        input_losses = []
        # tqdm(range(len(dataset)))
        print(len(dataset))
        for date_id in [len(dataset) - 1]:
            test_data, test_label = dataset[date_id]
            test_data, test_label = np.expand_dims(test_data, 0), np.expand_dims(test_label, 0)
            datefile_id, hour = dataset.get_path_id(date_id)
            date = dataset.wrf_files[datefile_id].split('_')[-2]
            test_data = torch.swapaxes(torch.from_numpy(test_data).type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            test_label = torch.swapaxes(torch.from_numpy(test_label).type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            test_data = wrf_scaler.channel_transform(test_data, 2)
            test_label = era_scaler.channel_transform(test_label, 2)
            output = model(test_data)
            x = output - test_data[:, :, :3]
            output = wrf_scaler.channel_inverse_transform(output, 2)
            test_data = wrf_scaler.channel_inverse_transform(test_data, 2)[:, :, :3]
            test_label = era_scaler.channel_inverse_transform(test_label, 2)
            for j, loss in enumerate(losses):
                test_loss = loss(test_data, output, test_label, logger)
                input_loss = loss(test_data, test_data, test_label, logger)
                test_losses.append(test_loss)
                input_losses.append(input_loss)
            fig, axs = plt.subplots(4, 4, dpi=800)
            imgs_to_draw = list(map(lambda x: torch.Tensor.numpy(torch.Tensor.cpu(x)),
                                    [test_data, output - test_data, output, test_label]))
            print(imgs_to_draw[0][:, :, 1].min(), imgs_to_draw[0][:, :, 1].max())
            print(imgs_to_draw[1][:, :, 1].min(), imgs_to_draw[1][:, :, 1].max())
            print(imgs_to_draw[2][:, :, 1].min(), imgs_to_draw[2][:, :, 1].max())
            print(imgs_to_draw[3][:, :, 1].min(), imgs_to_draw[3][:, :, 1].max())
            for i, title in enumerate(['Original WRF image', 'Correction', 'Corrected WRF image', 'ERA5 image']):
                axs[i][0].set_title(title)
                # vmin, vmax = (-1, 10) if i == 1 else (255, 290)
                vmin, vmax = (-10, 16) if i == 1 else (-10, 16)
                # print(vmin, vmax)
                for t in range(4):
                    # print(imgs_to_draw[i][t, 0, 2].min(), imgs_to_draw[i][t, 0, 2].max(), i, t)
                    im = axs[i][t].imshow(imgs_to_draw[i][t, 0, 1], extent=[0, 280, 0, 210], vmin=vmin, vmax=vmax)
                    axs[i][t].axison = False
            axs[i][0].text(10, 10, f'date: {date} {hour}:00', size='x-small')
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.86, 0.15, 0.03, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            plots_path = os.path.join(cfg.GLOBAL.BASE_DIR, logs_dir, f'plots/v10')
            if not os.path.exists(plots_path):
                os.makedirs(plots_path)
            fig.savefig(os.path.join(plots_path, f'mega_plot_{date}.png'))
            break

    return test_loss


def save_infernce(model, criterion, wrf_scaler, era_scaler, dataset, batch_size, logs_dir, logger=None,
                  draw_plots=False):
    iterator = range(0, len(dataset), batch_size)
    i = 0
    with torch.no_grad():
        model.eval()
        for test_data, test_label in tqdm(iterator):
            print(test_data.shape, test_label.shape)

            test_data = torch.swapaxes(test_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            test_label = torch.swapaxes(test_label.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            test_data = wrf_scaler.channel_transform(test_data, 2)
            test_label = era_scaler.channel_transform(test_label, 2)
            print(test_data.shape, test_label.shape)
            test_data = torch.split(test_data, 4, dim=0)
            test_label = torch.split(test_label, 4, dim=0)
            corr = []
            for i in range(len(test_data)):
                output = model(test_data[i])
                corr.append(output)

            test_label = torch.cat(test_label)
            test_label = era_scaler.channel_inverse_transform(test_label, 2)

            np.save(os.path.join(cfg.GLOBAL.BASE_DIR, logs_dir, 'wrf_corr_sample'), test_label)
            break
    return
