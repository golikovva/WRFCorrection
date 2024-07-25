from __future__ import division, print_function, absolute_import
from matplotlib.patches import Arc
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import torch
import os


def draw_borey_basemap(era_season_map, lats, lons, dtype='WRF', date='2019-01-01', channel='t2', colormap='common'):
    figs = {}
    fig, ax = plt.subplots(1, dpi=300)
    fig.suptitle(f'{dtype} ERA5 {channel} {date} difference map', fontsize=16)

    ax.set_xticks([])
    ax.set_yticks([])
    m = Basemap(
        projection='laea',
        resolution='i',
        lat_0=71.0,
        lon_0=80.0,
        llcrnrlon=42.27414,
        llcrnrlat=62.22973,
        urcrnrlon=72.22717,
        urcrnrlat=78.75766,

    )
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(50, 90, 2), labels=[1, 0, 0, 0], color='gray')
    m.drawmeridians(np.arange(-180, 180, 5), labels=[0, 0, 0, 1], color='gray')
    x, y = m(lons, lats)
    vmin, vmax, mean = np.nanmin(era_season_map), np.nanmax(era_season_map), np.nanmean(era_season_map)
    if colormap == 'common':
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = 'viridis'
    else:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cmap = 'bwr'
    img = m.pcolor(x, y, np.squeeze(era_season_map), cmap=cmap, norm=norm)
    # img = m.pcolormesh(lons, lats, np.squeeze(era_season_map), latlon=True, cmap='bwr')
    cbar = m.colorbar(img, location='bottom', pad="10%")
    fig.tight_layout()
    figs[f'{dtype.lower()}_era_{date}'] = fig
    return figs, ax, m


def borey_basemap_ax(data, lons, lats, colormap='common', ax=None):
    if ax is None:
        ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])
    m = Basemap(
        projection='laea',
        resolution='i',
        lat_0=71.0,
        lon_0=80.0,
        llcrnrlon=42.27414,
        llcrnrlat=62.22973,
        urcrnrlon=72.22717,
        urcrnrlat=78.75766,
        ax=ax,
    )
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(50, 90, 2), labels=[1, 0, 0, 0], color='gray')
    m.drawmeridians(np.arange(-180, 180, 5), labels=[0, 0, 0, 1], color='gray')
    x, y = m(lons, lats)
    vmin, vmax = np.nanmin(data), np.nanpercentile(data, 95, method='closest_observation')
    if colormap == 'common':
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = 'viridis'
    else:
        norm = colors.TwoSlopeNorm(vmin=min(-0.001, vmin), vcenter=0, vmax=vmax)
        cmap = 'bwr'
    img = m.pcolor(x, y, np.squeeze(data), cmap=cmap, norm=norm)
    cbar = m.colorbar(img, location='bottom', pad="10%")
    return ax


def draw_mega_plot(wrf_tensor, era_tensor, wrfcorr_tensor, channel, date, hour, era_metric=None, station_metric=None):
    fig, axs = plt.subplots(4, 4, dpi=600)
    imgs_to_draw = list(map(lambda x: torch.Tensor.numpy(torch.Tensor.cpu(x)),
                            [wrf_tensor, wrfcorr_tensor, era_tensor, wrfcorr_tensor - wrf_tensor]))

    for i, title in enumerate(['Original WRF image', 'Corrected WRF image', 'ERA5 image', 'Correction']):
        axs[i][0].set_title(title)
        if title == 'Correction':
            vmin, vmax = (imgs_to_draw[i][:, :, channel].min(), imgs_to_draw[i][:, :, channel].max())
        else:
            vmin, vmax = (imgs_to_draw[0][:, :, channel].min() - 5, imgs_to_draw[0][:, :, channel].max() + 5)
        for t in range(4):
            im = axs[i][t].imshow(imgs_to_draw[i][t, 0, channel], interpolation='none',
                                  extent=[0, 280, 0, 210], vmin=vmin, vmax=vmax)
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


def draw_station_err_map(metadata, wrf_sample, station_err_map):
    t2_err = station_err_map[:, 0]
    w10_err = station_err_map[:, 1]
    fig, ax = plt.subplots(dpi=800)
    cmap = plt.cm.get_cmap('Reds')
    norm = colors.Normalize(vmin=0, vmax=1.3)

    plt.scatter(metadata['coords'][:, 0], metadata['coords'][:, 1], label='t2/w10 station metric', s=0.2)
    plt.pcolormesh(metadata['wrf_xx'], metadata['wrf_yy'], wrf_sample, shading='auto')

    theta1, theta2 = 90, 90 + 180
    radius = 0.2

    arcs(metadata['coords'][:, 0], metadata['coords'][:, 1], 3 * radius, radius, theta1=theta1, theta2=theta2,
         color=cmap(norm(t2_err.cpu().numpy())))
    arcs(metadata['coords'][:, 0], metadata['coords'][:, 1], 3 * radius, radius, theta1=theta2, theta2=theta1,
         color=cmap(norm(w10_err.cpu().numpy())))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.legend()

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', fraction=0.1)


def draw_era_error_map(era_err_map):
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    for i in range(3):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        im1 = axs[i].imshow(era_err_map[i].reshape(67, 215), interpolation='none',
                            extent=[0, 280, 0, 210], )  # vmin=0, vmax=0.7)
        # im2 = axs[1].imshow(era_err_map[2].reshape(67, 215), interpolation='none', extent=[0, 280, 0, 210], vmin=0,
        #                     vmax=2.9)
        fig.colorbar(im1, ax=axs[i], orientation='vertical', fraction=0.1)
    return fig, axs


def draw_seasonal_era_error_map(era_season_map, dtype='WRF'):
    figs = {}
    for c, channel in enumerate(['u10', 'v10', 't2']):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'{dtype} ERA5 {channel} seasonal error maps', fontsize=16)
        for t, season in enumerate(['Winter', 'Spring', 'Summer', 'Autumn']):
            row, col = t // 2, t % 2
            axs[row][col].set_xticks([])
            axs[row][col].set_yticks([])
            vmin, vmax = 0, era_season_map[:, c].max()
            im1 = axs[row][col].imshow(era_season_map[t, c].reshape(67, 215), interpolation='none',
                                       extent=[0, 280, 0, 210], vmin=vmin, vmax=vmax)
            axs[row][col].set_title(season)
            divider = make_axes_locatable(axs[row][col])
            colorbar_axes = divider.append_axes("right", size="10%", pad=0.1)
            fig.colorbar(im1, cax=colorbar_axes)

        figs[f'{dtype.lower()}_era_seasonal_{channel}'] = fig

        fig, ax = plt.subplots(1)
        fig.suptitle(f'{dtype} ERA5 {channel} error map', fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        vmin, vmax = 0, era_season_map.mean(0)[c].max()
        im1 = ax.imshow(era_season_map.mean(0)[c].reshape(67, 215), interpolation='none',
                        extent=[0, 280, 0, 210], vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        colorbar_axes = divider.append_axes("right", size="10%", pad=0.1)
        fig.colorbar(im1, cax=colorbar_axes)
        figs[f'{dtype.lower()}_era_{channel}'] = fig
    return figs


def draw_seasonal_stations_error_map(station_season_map, metadata, wrf_sample, era_sample, dtype='WRF'):
    figs = {}
    for i, s in enumerate(['Winter', 'Spring', 'Summer', 'Autumn']):
        f = draw_station_metrics(metadata, era_sample, wrf_sample, station_season_map[i, 0], station_season_map[i, 1],
                                 title=f'{s} {dtype} stations t2,w10 metric')
        figs[f'{s.lower()}_{dtype.lower()}-stations-metric'] = f
    mean_map = station_season_map.mean(0)
    f = draw_station_metrics(metadata, era_sample, wrf_sample, mean_map[0], mean_map[1],
                             title=f'{dtype} stations t2,w10 metric')
    figs[f'mean_{dtype.lower()}-stations-metric'] = f
    return figs


def draw_scat_err_map(scat_err_map, lons, lats, title='Scatterometer error map', colormap='common'):
    fig, axs = plt.subplots(2, dpi=300, figsize=(10, 10))
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title('U10')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title('V10')
    borey_basemap_ax(scat_err_map[0].reshape(132, 430), lons=lons, lats=lats, ax=axs[0], colormap=colormap)
    borey_basemap_ax(scat_err_map[0].reshape(132, 430), lons=lons, lats=lats, ax=axs[1], colormap=colormap)
    # im1 = axs[0].imshow(scat_err_map[0].reshape(132, 430), interpolation='none', extent=[0, 280, 0, 210], )
    # im2 = axs[1].imshow(scat_err_map[1].reshape(132, 430), interpolation='none', extent=[0, 280, 0, 210], )
    # fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.1)
    # fig.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.1)
    fig.suptitle(title)
    return fig


def draw_seasonal_scat_err_map(seasonal_scat, lons, lats, dtype='WRF', colormap='common'):
    figs = {}
    for i, s in enumerate(['Winter', 'Spring', 'Summer', 'Autumn']):
        f = draw_scat_err_map(seasonal_scat[i], lons, lats, title=f'{s} {dtype} scatterometer error map', colormap=colormap)
        figs[f'{s.lower()}_{dtype.lower()}_scatter_error_map'] = f
    f = draw_scat_err_map(seasonal_scat[4], lons, lats, title=f'{dtype} scatterometer error map', colormap=colormap)
    figs[f'mean_{dtype.lower()}_scatter_error_map'] = f
    return figs


def draw_simple_plots(wrf_tensor, wrfcorr_tensor, era_tensor, channel=2,
                      input_loss=-1, test_loss=-1, era_metric=None, station_metric=None, date=None):
    fig, axs = plt.subplots(4, 1, figsize=(5, 14))
    vmin = min(wrf_tensor[:, :, channel].min(), wrfcorr_tensor[:, :, channel].min(), era_tensor[:, :, channel].min())
    vmax = max(wrf_tensor[:, :, channel].max(), wrfcorr_tensor[:, :, channel].max(), era_tensor[:, :, channel].max())
    im = axs[0].imshow(wrf_tensor[0, 0, channel].cpu().numpy(), interpolation='none', vmin=vmin, vmax=vmax)
    axs[1].imshow(wrfcorr_tensor[0, 0, channel].cpu().numpy(), interpolation='none', vmin=vmin, vmax=vmax)
    axs[2].imshow(era_tensor[0, 0, channel].cpu().numpy(), interpolation='none',
                  extent=[0, 280, 0, 210], vmin=vmin, vmax=vmax)
    imc = axs[3].imshow(wrfcorr_tensor[0, 0, channel].cpu().numpy() - wrf_tensor[0, 0, channel].cpu().numpy(),
                        interpolation='none', extent=[0, 280, 0, 210], )
    axs[0].set_xlabel('Original WRF data')
    axs[0].text(50, 28, f'loss={round(input_loss, 3)}')
    axs[1].set_xlabel('Corrected WRF data')
    axs[1].text(50, 28, f'loss={round(test_loss, 3)}')
    axs[2].set_xlabel('ERA5 reanalysis')
    axs[3].set_xlabel('Correction')
    for i in range(4):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].xaxis.set_label_coords(.5, -.01)
    if era_metric:
        axs[3].text(0, -85, f'era metric={round(era_metric, 3)}')
    if station_metric:
        axs[3].text(0, -60, f'station metric={round(station_metric, 3)}')
    if date:
        axs[3].text(0, 5, f'{date}')
    fig.colorbar(im, ax=axs[:3], orientation='vertical', fraction=0.1, aspect=21)
    fig.colorbar(imc, ax=axs[3], orientation='vertical', fraction=0.1, aspect=6)

    return fig, axs


def draw_station_metrics(metadata, era_sample, wrf_sample, stations_t2_metric, stations_w10_metric,
                         title='Stations metrics'):
    fig, ax = plt.subplots(dpi=400)
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
    plt.title(title)
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.legend()

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', fraction=0.1)
    return fig


def draw_losses_gist(orig, corr, dtype, channels=None):
    channels = ['u10', 'v10', 't2'] if channels is None else channels
    fig, axs = plt.subplots(1, len(channels), figsize=(len(channels) * 4, 4))
    for i, channel in enumerate(channels):
        axs[i].hist(orig[i], bins=50, label='original')
        axs[i].hist(corr[i], bins=50, label='corrected')
        axs[i].set_xlabel(f'{dtype} {channel} losses')
    plt.legend()
    axs[0].set_ylabel('Num samples')
    fig.suptitle('Loss distribution')
    return fig, axs


def draw_seasonal_bar_plot(metric_mean, channels=None, dtype="ERA5"):
    channels = ['u10', 'v10', 't2'] if channels is None else channels
    fig, axs = plt.subplots(1)
    width = 0.25
    ind = np.arange(4)
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    for i, channel in enumerate(channels):
        axs.bar(ind + width * i, metric_mean[:, i], width, label=channel)

    axs.set_xlabel("Dates")
    axs.set_ylabel('Metric')
    axs.set_title(f"Seasonal {dtype} metric by channel")

    axs.set_xticks(ind + width * (len(channels) / 2 - 0.5), seasons)
    plt.legend()
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
