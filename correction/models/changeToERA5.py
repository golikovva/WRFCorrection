import time

import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np
from correction.helpers.interpolation import get_nearest_neighbour
# from correction.config.config import cfg
from correction.helpers.math import gauss_function


class MeanToERA5(nn.Module):
    def __init__(self, mapping_file=None, era_coords=None, wrf_coords=None, mapping_mode='kmeans',
                 weighted=False, device='cpu'):
        super().__init__()
        self.mapping = None
        self.indices = None
        self.distances = None
        self.reverse_distance = None
        self.weighted = weighted
        self.device = device
        if era_coords is not None and wrf_coords is not None:
            self.era_coords = era_coords.copy(order='C')
            self.wrf_coords = wrf_coords.copy(order='C')
        if mapping_file is not None:
            self.set_mapping_by_file(mapping_file)
        else:
            assert era_coords is not None and wrf_coords is not None
            if mapping_mode == 'kmeans':
                self.set_mapping_by_coords()
            if mapping_mode == 'nearest':
                self.set_mapping_by_nearest_neighbour()
        if weighted:
            self.calc_distances_by_coords()

    def set_mapping_by_file(self, mapping_file):
        self.mapping = torch.from_numpy(np.load(mapping_file))
        _, self.indices = self.mapping.sort(stable=True)

    def set_mapping_by_coords(self):
        n_clusters = self.era_coords.shape[0]
        kmeans = KMeans(n_clusters=n_clusters, n_init=1, max_iter=1)
        kmeans.fit(np.zeros([n_clusters, 2]))  # redundant action, needed to initialise kmeans internal params
        kmeans.cluster_centers_ = self.era_coords
        self.wrf_coords = self.wrf_coords
        self.mapping = torch.from_numpy(kmeans.predict(self.wrf_coords))
        _, self.indices = self.mapping.sort(stable=True)

    def set_mapping_by_nearest_neighbour(self):
        mapping, distances = get_nearest_neighbour(torch.from_numpy(self.wrf_coords).T,
                                                   torch.from_numpy(self.era_coords).T)
        self.mapping, self.distances = torch.squeeze(mapping), torch.squeeze(distances)
        _, self.indices = self.mapping.sort(stable=True)

    def calc_distances_by_coords(self):
        p1, p2 = self.wrf_coords, self.era_coords[self.mapping]
        self.distances = torch.from_numpy(np.sqrt(np.square(p1 - p2).sum(-1)))
        self.reverse_distance = (1 / self.distances).to(self.device)
        # sigma_squared = torch.square(self.distances.max())/9  # max element lies in 3*sigma
        # self.reverse_distance = gauss_function(self.distances, sigma_squared)).to(cfg.GLOBAL.DEVICE)

    def forward(self, output):
        if self.weighted:
            return self.forward_weighted(output)

        output = output.view(*output.shape[:-2], output.shape[-1] * output.shape[-2])
        a = []
        for lst in output[..., self.indices].split(tuple(self.mapping.unique(return_counts=True)[1]), dim=-1):
            a.append(lst.mean(-1))
        return torch.stack(a, dim=-1)

    def forward_weighted(self, output):
        output = output.flatten(-2, -1)
        a = []
        t1 = time.time()
        clusters = output[..., self.indices].split(tuple(self.mapping.unique(return_counts=True)[1]), dim=-1)
        # print(time.time() - t1, 'Time spent to calc clusters')
        t2 = time.time()
        reverse_distances = self.reverse_distance[self.indices].split(
            tuple(self.mapping.unique(return_counts=True)[1]), dim=-1)
        # print(time.time() - t2, 'Time spent to calc reverse distances')
        t3 = time.time()
        # todo оптимизировать следующую операцию, она занимает большую часть (чуть ли не половину) цикла обучения!!!
        for cluster, reverse_distance in zip(clusters, reverse_distances):
            weighted_meaned_value = (reverse_distance * cluster).sum(-1) / reverse_distance.sum()
            a.append(weighted_meaned_value)
        # print(time.time() - t3, 'Time spent to multiply cluster on rev dist')
        return torch.stack(a, dim=-1)

    def save_mapping(self, filename):
        np.save(filename, self.mapping)

    def to(self, device):
        self.device = device
        if self.reverse_distance is not None:
            self.reverse_distance = self.reverse_distance.to(device)
        return self


if __name__ == '__main__':
    from osgeo import gdal

    wrf_out_file = "C:\\Users\\Viktor\\Desktop\\wrfout_d01_2019-01-01_000000"

    ds_lon = gdal.Open('NETCDF:"' + wrf_out_file + '":XLONG')
    print("lon", ds_lon.ReadAsArray().shape)

    ds_lat = gdal.Open('NETCDF:"' + wrf_out_file + '":XLAT')
    print("ds_lat", ds_lat.ReadAsArray().shape)
    lat1, lat2 = ds_lat.ReadAsArray().min(), ds_lat.ReadAsArray().max()
    lon1, lon2 = ds_lon.ReadAsArray().min(), ds_lon.ReadAsArray().max()

    era_out_file = "C:\\Users\\Viktor\\Desktop\\ERA5_uv10m_2019-01.nc"

    era_lat = gdal.Open('NETCDF:"' + era_out_file + '":latitude')
    print('era_lat', era_lat.ReadAsArray().shape)
    era_lon = gdal.Open('NETCDF:"' + era_out_file + '":longitude')
    print('era_lon', era_lon.ReadAsArray().shape)

    cond1 = (era_lat.ReadAsArray() > lat1) & (era_lat.ReadAsArray() < lat2)
    cond2 = (era_lon.ReadAsArray() > lon1) & (era_lon.ReadAsArray() < lon2)
    xx, yy = np.meshgrid(era_lon.ReadAsArray()[cond2], era_lat.ReadAsArray()[cond1])
    era5_cluster_centers = np.stack([xx, yy], 0).reshape(2, -1).T

    wrf_coords = np.stack([ds_lon.ReadAsArray()[0], ds_lat.ReadAsArray()[0]], 0).reshape(2, -1).T

    # meaner = MeanToERA5(era_coords=era5_cluster_centers, wrf_coords=wrf_coords)
    meaner = MeanToERA5('C:\\Users\\Viktor\\ml\\Precipitation-Nowcasting-master\\wrferaMapping.npy')

    a = torch.rand(2, 1, 210, 280)
    out = meaner.forward(a)
    print(out.shape)

    meaner.save_mapping('wrferaMapping')
