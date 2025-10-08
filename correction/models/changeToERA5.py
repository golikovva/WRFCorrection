import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np
from correction.helpers.interpolation import get_nearest_neighbour
from correction.helpers.math import gauss_function
from sklearn.neighbors import NearestNeighbors



class ClusterMapper(nn.Module):
    def __init__(self, mapping_file=None, target_coords=None, input_coords=None, 
                 weighted=False, save_mapping=False, save_name='meaner_mapping.npy', 
                 device='cpu', distance_metric='euclidean'):
        super().__init__()
        self.mapping = None
        self.distances = None
        self.reverse_distance = None
        self.denominator = None
        self.counts = None
        self.mask = None
        self.weighted = weighted
        self.device = device
        self.distance_metric = distance_metric

        if target_coords is not None and input_coords is not None:
            self.target_coords = target_coords.copy(order='C')
            self.input_coords = input_coords.copy(order='C')

        if mapping_file is not None:
            self.set_mapping_by_file(mapping_file)
        else:
            assert target_coords is not None and input_coords is not None
            self.create_mapping()
            if save_mapping:
                self.save_mapping(save_name)
        self._precompute_counts_mask()

        if self.weighted:
            self.calc_weights()

    def set_mapping_by_file(self, mapping_file):
        data = np.load(mapping_file, allow_pickle=True).item()
        self.mapping = torch.from_numpy(data['indices']).long()
        if 'distances' in data:
            self.distances = torch.from_numpy(data['distances'])

    def create_mapping(self):
        """Unified method for creating mapping using NearestNeighbors"""
        if self.distance_metric == 'haversine':
            target_coords = np.radians(self.target_coords)
            input_coords = np.radians(self.input_coords)
        else:
            target_coords = self.target_coords
            input_coords = self.input_coords

        nearn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric=self.distance_metric)
        nearn.fit(target_coords)
        distances, indices = nearn.kneighbors(input_coords)
        
        self.mapping = torch.from_numpy(indices.squeeze()).long()
        self.distances = torch.from_numpy(distances.squeeze())

    def _precompute_counts_mask(self):
        self.counts = torch.bincount(self.mapping, minlength=self.target_coords.shape[0])
        self.mask = self.counts > 0

    def calc_weights(self):
        """Use precomputed distances from NearestNeighbors"""
        if self.distances is None:
            raise RuntimeError("Distances not calculated. Call create_mapping first.")

        self.reverse_distance = (1 / self.distances).to(self.device)
        self.denominator = torch.zeros(self.target_coords.shape[0], 
                                     device=self.device,
                                     dtype=self.reverse_distance.dtype)
        self.denominator.scatter_add_(0, self.mapping.to(self.device), self.reverse_distance)
        self.denominator = self.denominator.clamp(min=1e-6)

    def save_mapping(self, filename):
        data = {
            'indices': self.mapping.numpy(),
            'distances': self.distances.numpy() if self.distances is not None else None
        }
        np.save(filename, data)

    def forward(self, output, masked=True):
        if self.weighted:
            res =  self._forward_weighted(output)
        else:
            res = self._forward_mean(output)
        if masked:
            res = res[..., self.mask.to(output.device)]
        return res

    def _forward_mean(self, output):
        output_flat = output.flatten(-2, -1)
        mapping = self.mapping.expand_as(output_flat).to(output.device)
        summed = torch.zeros(*output_flat.shape[:-1], self.target_coords.shape[0],
                           device=output.device)
        summed.scatter_add_(-1, mapping, output_flat)
        return (summed / self.counts.to(output.device).clamp(min=1e-6))

    def _forward_weighted(self, output):
        output_flat = output.flatten(-2, -1)
        reverse_distance = self.reverse_distance.view(*([1]*(output_flat.dim()-1)), -1)
        weighted_output = output_flat * reverse_distance.to(output.device, dtype=output.dtype)
        mapping = self.mapping.expand_as(weighted_output).to(output.device)
        summed = torch.zeros(*weighted_output.shape[:-1], self.target_coords.shape[0],
                           device=output.device, dtype=output.dtype)
        summed.scatter_add_(-1, mapping, weighted_output)
        return (summed / self.denominator.to(output.device, dtype=output.dtype).clamp(min=1e-6))

    def to(self, device):
        self.device = device
        for attr in ['reverse_distance', 'denominator', 'mapping', 'counts', 'mask', 'distances']:
            tensor = getattr(self, attr)
            if tensor is not None:
                setattr(self, attr, tensor.to(device))
        return self
    

class MeanToERA5(nn.Module):
    def __init__(self, mapping_file=None, era_coords=None, wrf_coords=None, mapping_mode='kmeans',
                 weighted=False, device='cpu'):
        super().__init__()
        self.mapping = None
        self.indices = None
        self.distances = None
        self.reverse_distance = None
        self.denominator = None
        self.counts = None
        self.mask = None
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
            elif mapping_mode == 'nearest':
                self.set_mapping_by_nearest_neighbour()
        
        if weighted:
            self.calc_distances_by_coords()
        else:
            self._precompute_counts_mask()

    def set_mapping_by_file(self, mapping_file):
        self.mapping = torch.from_numpy(np.load(mapping_file)).long()
        _, self.indices = self.mapping.sort(stable=True)
        self._precompute_counts_mask()

    def set_mapping_by_coords(self):
        n_clusters = self.era_coords.shape[0]
        kmeans = KMeans(n_clusters=n_clusters, n_init=1, max_iter=1)
        kmeans.fit(np.zeros([n_clusters, 2]))
        kmeans.cluster_centers_ = self.era_coords
        self.mapping = torch.from_numpy(kmeans.predict(self.wrf_coords.astype(float))).long()
        self._precompute_counts_mask()

    def _precompute_counts_mask(self):
        if self.mapping is not None:
            self.counts = torch.bincount(self.mapping, minlength=self.era_coords.shape[0])
            self.mask = self.counts > 0

    def calc_distances_by_coords(self):
        p1, p2 = self.wrf_coords, self.era_coords[self.mapping]
        self.distances = torch.from_numpy(np.sqrt(np.square(p1 - p2).sum(-1)))
        self.reverse_distance = (1 / self.distances).to(self.device)
        self.denominator = torch.zeros(self.era_coords.shape[0], device=self.device, dtype=self.reverse_distance.dtype)
        self.denominator.scatter_add_(0, self.mapping.to(self.device), self.reverse_distance)
        self.denominator = self.denominator.clamp(min=1e-6)
        self._precompute_counts_mask()

    def forward(self, output, apply_mask=True):
        if self.weighted:
            return self.forward_weighted(output, apply_mask)
        return self._forward_mean(output, apply_mask)

    def _forward_mean(self, output, apply_mask):
        output_flat = output.flatten(-2, -1)
        mapping = self.mapping.expand_as(output_flat).to(output.device)
        summed = torch.zeros(*output_flat.shape[:-1], self.era_coords.shape[0], device=output.device)
        summed.scatter_add_(-1, mapping, output_flat)
        mean = summed / self.counts.to(output.device).clamp(min=1e-6)
        return mean[..., self.mask.to(output.device)] if apply_mask else mean

    def forward_weighted(self, output, apply_mask):
        output_flat = output.flatten(-2, -1)
        reverse_distance = self.reverse_distance.view(*([1]*(output_flat.dim()-1)), -1).to(output.device, dtype=output.dtype)
        weighted_output = output_flat * reverse_distance
        mapping = self.mapping.expand_as(weighted_output).to(output.device)
        summed = torch.zeros(*weighted_output.shape[:-1], self.era_coords.shape[0], device=output.device, dtype=output.dtype)
        summed.scatter_add_(-1, mapping, weighted_output)
        weighted_mean = summed / self.denominator.to(output.device).clamp(min=1e-6)
        return weighted_mean[..., self.mask.to(output.device)] if apply_mask else weighted_mean

    def to(self, device):
        self.device = device
        tensors = ['reverse_distance', 'denominator', 'mapping', 'counts', 'mask']
        for attr in tensors:
            tensor = getattr(self, attr)
            if tensor is not None:
                setattr(self, attr, tensor.to(device))
        return self
    

class MeanToERA5_old(nn.Module):
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


# if __name__ == '__main__':
#     from osgeo import gdal

#     wrf_out_file = "C:\\Users\\Viktor\\Desktop\\wrfout_d01_2019-01-01_000000"

#     ds_lon = gdal.Open('NETCDF:"' + wrf_out_file + '":XLONG')
#     print("lon", ds_lon.ReadAsArray().shape)

#     ds_lat = gdal.Open('NETCDF:"' + wrf_out_file + '":XLAT')
#     print("ds_lat", ds_lat.ReadAsArray().shape)
#     lat1, lat2 = ds_lat.ReadAsArray().min(), ds_lat.ReadAsArray().max()
#     lon1, lon2 = ds_lon.ReadAsArray().min(), ds_lon.ReadAsArray().max()

#     era_out_file = "C:\\Users\\Viktor\\Desktop\\ERA5_uv10m_2019-01.nc"

#     era_lat = gdal.Open('NETCDF:"' + era_out_file + '":latitude')
#     print('era_lat', era_lat.ReadAsArray().shape)
#     era_lon = gdal.Open('NETCDF:"' + era_out_file + '":longitude')
#     print('era_lon', era_lon.ReadAsArray().shape)

#     cond1 = (era_lat.ReadAsArray() > lat1) & (era_lat.ReadAsArray() < lat2)
#     cond2 = (era_lon.ReadAsArray() > lon1) & (era_lon.ReadAsArray() < lon2)
#     xx, yy = np.meshgrid(era_lon.ReadAsArray()[cond2], era_lat.ReadAsArray()[cond1])
#     era5_cluster_centers = np.stack([xx, yy], 0).reshape(2, -1).T

#     wrf_coords = np.stack([ds_lon.ReadAsArray()[0], ds_lat.ReadAsArray()[0]], 0).reshape(2, -1).T

#     # meaner = MeanToERA5(era_coords=era5_cluster_centers, wrf_coords=wrf_coords)
#     meaner = MeanToERA5('C:\\Users\\Viktor\\ml\\Precipitation-Nowcasting-master\\wrferaMapping.npy')

#     a = torch.rand(2, 1, 210, 280)
#     out = meaner.forward(a)
#     print(out.shape)

#     meaner.save_mapping('wrferaMapping')
