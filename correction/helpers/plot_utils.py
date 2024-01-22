from __future__ import division, print_function, absolute_import
from matplotlib.patches import Arc
from matplotlib.collections import PatchCollection

import matplotlib
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


def draw_station_metrics(metadata, era_sample, wrf_sample, stations_t2_metric, stations_w10_metric):
    fig, ax = plt.subplots(dpi=800)
    cmap = plt.cm.get_cmap('bwr_r')
    norm = colors.Normalize(vmin=-1, vmax=1)

    plt.scatter(metadata['coords'][:, 0], metadata['coords'][:, 1], label='t2/w10 station metric')
    plt.pcolormesh(metadata['era_xx'], metadata['era_yy'], era_sample[0, 0, 2].cpu().numpy(), shading='auto')

    plt.pcolormesh(metadata['wrf_xx'], metadata['wrf_yy'], wrf_sample[0, 0, 2].cpu().numpy(), shading='auto')

    theta1, theta2 = 90, 90 + 180
    radius = 0.2

    arcs(metadata['coords'][:, 0], metadata['coords'][:, 1], 3 * radius, radius, theta1=theta1, theta2=theta2,
         color=cmap(norm(stations_t2_metric.cpu().numpy())))
    arcs(metadata['coords'][:, 0], metadata['coords'][:, 1], 3 * radius, radius, theta1=theta2, theta2=theta1,
         color=cmap(norm(stations_w10_metric.cpu().numpy())))
    plt.title('Местоположение станций')
    plt.xlabel('Широта')
    plt.ylabel('Долгота')
    plt.legend()

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', fraction=0.1)


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
        t2_stations_metrics = losses_df.loc[
            losses_df['wrf_dates'].apply(get_season) == season, 't2_stations_metrics'].mean()
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


def arcs(x, y, w, h=None, rot=0.0, theta1=0.0, theta2=360.0,
         c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter plot of Arcs.
    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Center of ellipses.
    w, h : scalar or array_like, shape (n, )
        Total length (diameter) of horizontal/vertical axis.
        `h` is set to be equal to `w` by default, ie. circle.
    rot : scalar or array_like, shape (n, )
        Rotation in degrees (anti-clockwise).
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.
    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
    Examples
    --------
    a = np.arange(11)
    arcs(a, a, w=4, h=a, rot=a*30, theta1=0.0, theta2=180.0,
         c=a, alpha=0.5, ec='none')
    plt.colorbar()
    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    if h is None:
        h = w

    zipped = np.broadcast(x, y, w, h, rot, theta1, theta2)
    patches = [Arc((x_, y_), w_, h_, angle=rot_, theta1=t1_, theta2=t2_)
               for x_, y_, w_, h_, rot_, t1_, t2_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection
