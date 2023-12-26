import matplotlib.pyplot as plt
from correction.config.config import cfg
import numpy as np
from matplotlib import colors
import torch
import os


def draw_mega_plot(wrf_tensor, era_tensor, wrfcorr_tensor, channel, date, hour, era_metric=None, station_metric=None):
    out_len = cfg.HKO.BENCHMARK.OUT_LEN
    fig, axs = plt.subplots(4, out_len)
    imgs_to_draw = list(map(lambda x: torch.Tensor.numpy(torch.Tensor.cpu(x)),
                            [wrf_tensor, wrfcorr_tensor, era_tensor, wrfcorr_tensor - wrf_tensor]))

    for i, title in enumerate(['Original WRF image', 'Corrected WRF image', 'ERA5 image', 'Correction']):
        axs[i][0].set_title(title)
        if title == 'Correction':
            vmin, vmax = (imgs_to_draw[i][:, :, channel].min(), imgs_to_draw[i][:, :, channel].max())
        else:
            vmin, vmax = (imgs_to_draw[0][:, :, channel].min() - 5, imgs_to_draw[0][:, :, channel].max() + 5)
        for t in range(out_len):
            im = axs[i][t].imshow(imgs_to_draw[i][t, 0, channel], extent=[0, 280, 0, 210], vmin=vmin, vmax=vmax)
            axs[i][t].axison = False
        if title == 'Corrected WRF image':
            fig.colorbar(im, ax=axs[:3], orientation='vertical', fraction=0.1, aspect=21)
        elif title == 'Correction':
            fig.colorbar(im, ax=axs[3], orientation='vertical', fraction=0.1, aspect=7)
    if era_metric:
        axs[i][1].text(10, 10, f'era metric={round(era_metric, 3)}', size='xx-small')
    if station_metric:
        axs[i][2].text(10, 10, f'station metric={round(station_metric, 3)}', size='xx-small')
    axs[i][0].text(10, 10, f'date: {date} {hour}:00', size='xx-small')


def draw_simple_plots(wrf_tensor, wrfcorr_tensor, era_tensor, channel=2,
                      input_loss=None, test_loss=None, era_metric=None, station_metric=None, date=None):
    fig, axs = plt.subplots(3, 1, dpi=800)
    vmin = min(wrf_tensor[:, :, channel].min(), wrfcorr_tensor[:, :, channel].min(), era_tensor[:, :, channel].min())
    vmax = max(wrf_tensor[:, :, channel].max(), wrfcorr_tensor[:, :, channel].max(), era_tensor[:, :, channel].max())
    im = axs[0].imshow(wrf_tensor[0, 0, channel].cpu().numpy(), interpolation='none', vmin=vmin, vmax=vmax)
    axs[1].imshow(wrfcorr_tensor[0, 0, channel].cpu().numpy(), interpolation='none', vmin=vmin, vmax=vmax)
    axs[2].imshow(era_tensor[0, 0, channel].cpu().numpy(), interpolation='none',
                  extent=[0, 280, 0, 210], vmin=vmin, vmax=vmax)
    axs[0].set_xlabel('Original WRF data')
    axs[0].text(50, 28, f'loss={round(input_loss, 3)}')
    axs[1].set_xlabel('Corrected WRF data')
    axs[1].text(50, 28, f'loss={round(test_loss, 3)}')
    axs[2].set_xlabel('ERA5 reanalysis')
    for i in range(3):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].xaxis.set_label_coords(.5, -.01)
    if era_metric:
        axs[2].text(0, -85, f'era metric={round(era_metric, 3)}')
    if station_metric:
        axs[2].text(0, -60, f'station metric={round(station_metric, 3)}')
    if date:
        axs[2].text(0, 5, f'{date}')
    fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.1)
    return fig, axs


def make_metric_gist(losses_df):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].hist(losses_df['metrics'], bins=50)
    axs[1].hist(losses_df['t2_stations_metrics'], bins=50)
    axs[2].hist(losses_df['w10_stations_metrics'], bins=50)

    axs[0].set_xlabel('Metric value on ERA5')
    axs[0].set_ylabel('Num samples')
    axs[1].set_xlabel('Station t2 metric value')
    axs[2].set_xlabel('Station wspd10 metric value')
    return fig, axs


def draw_season_metric_plot(losses_df):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    def get_season(data):
        month = int(data.split('-')[-2])
        return month // 3 % 4

    losses_df['wrf_dates'].apply(get_season)
    season_losses = []
    t2_season_station_losses, w10_season_station_losses = [], []
    for season in range(4):
        loss = losses_df.loc[losses_df['wrf_dates'].apply(get_season) == season, 'metrics'].mean()
        t2_stations_metrics = losses_df.loc[losses_df['wrf_dates'].apply(get_season) == season, 't2_stations_metrics'].mean()
        w10_stations_metrics = losses_df.loc[
            losses_df['wrf_dates'].apply(get_season) == season, 'w10_stations_metrics'].mean()
        season_losses.append(loss)
        t2_season_station_losses.append(t2_stations_metrics)
        w10_season_station_losses.append(w10_stations_metrics)
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    axs[0].bar(seasons, season_losses, edgecolor='black')
    axs[1].bar(seasons, t2_season_station_losses, edgecolor='black')
    axs[2].bar(seasons, w10_season_station_losses, edgecolor='black')

    axs[0].set_ylabel('Mean metric')
    axs[0].set_xlabel('Seasonal metric on ERA5')
    axs[1].set_xlabel('Seasonal t2 station metric')
    axs[2].set_xlabel('Seasonal w10 station metric')
    return fig, axs
