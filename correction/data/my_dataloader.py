import numpy as np
# import pandas as pd
# import torch
from torch.utils.data import DataLoader, Dataset, default_collate, Sampler
from correction.config.cfg import cfg
from correction.helpers.interpolation import create_mask_by_nearest_to_nans
from correction.helpers.interpolation import InvDistTree
import time
# import os
import netCDF4
import wrf
import pickle
import os
import pendulum as pdl
import xarray as xr


class WRFDataset(Dataset):
    def __init__(self, wrf_files, era_files, station_files=None, scatter_files=None,
                 wrf_variables=None, era_variables=None, seq_len=4,
                 use_spatial_encoding=False, use_time_encoding=False, use_landmask=False):
        super().__init__()

        print(len(wrf_files), 'wrf_files length')
        print(len(era_files), 'era_files length')
        wrf_files.sort()
        era_files.sort()
        self.wrf_files = wrf_files
        self.era_files = era_files

        if era_variables is None:
            era_variables = ['u10', 'v10', 't2m']
        self.era_variables = era_variables
        if wrf_variables is None:
            wrf_variables = ['uvmet10', 'T2']
        if use_spatial_encoding:
            wrf_variables = wrf_variables + ['XLAT', 'XLONG']
        if use_time_encoding:
            self.create_wrf_meshgrid()
        self.landmask = None
        if use_landmask:
            self.landmask = self.create_landmask_from_sample(self.wrf_files[0])
        self.wrf_variables = wrf_variables

        self.metadata = {}
        self.load_grid_metadata()
        self.stations = None

        if station_files is not None:
            self.load_stations(station_files)

        self.scatter_data = None
        if scatter_files is not None:
            self.load_scatters(scatter_files)

        self.seq_len = seq_len
        self.file_len = 24

        self.use_landmask = use_landmask
        self.use_time_encoding = use_time_encoding
        self.use_spatiotemporal_encoding = use_spatial_encoding

    @staticmethod
    def create_landmask_from_sample(sample_filename):
        with xr.open_dataset(sample_filename) as sample:
            land_mask = np.flip(~(sample['HGT'].data[0] < 5e-1)*sample['LANDMASK'].data[0], 0)
        return land_mask

    def create_wrf_meshgrid(self):
        x, y = np.arange(0, 280, 1), np.arange(0, 210, 1)
        self.wrf_xy = np.meshgrid(x, y)

    def get_date_by_datefile_id(self, datefile_id, np_like=False):
        if np_like:
            return np.datetime64(self.wrf_files[datefile_id].split('_')[-2])
        return pdl.parse(self.wrf_files[datefile_id].split('_')[-2])

    def get_day_hour_encoding(self, i, frequency=1):
        datefile_id, hour = self.get_path_id(i)
        date = self.get_date_by_datefile_id(datefile_id)
        day = [date.add(hours=i).day_of_year for i in range(self.seq_len)]
        day = np.array(day) / 365
        hour = np.mod(np.array(range(hour, hour + self.seq_len)), 24) / 24
        d1 = abs(abs(0.5 - day) - 0.5) + 0.05
        d2 = abs(abs(0.25 - day) - 0.5) + 0.05
        h1 = abs(abs(0.5 - hour) - 0.5) + 0.05
        h2 = abs(abs(0.25 - hour) - 0.5) + 0.05
        day_encoded = (np.sin(frequency * np.einsum('i,jk->ijk', d1, self.wrf_xy[0]))
                       + np.sin(frequency * np.einsum('i,jk->ijk', d2, self.wrf_xy[1]))) / 2
        hour_encoded = (np.cos(frequency * np.einsum('i,jk->ijk', h1, self.wrf_xy[0]))
                        + np.cos(frequency * np.einsum('i,jk->ijk', h2, self.wrf_xy[1]))) / 2
        return day_encoded, hour_encoded

    @staticmethod
    def get_time_encoding(date_normalized, lon, lat, frequency=4, periodic_law=np.sin):
        # day = (date.astype('datetime64[D]') - date.astype('datetime64[Y]')).astype(int) / 365
        if date_normalized.ndim == 0:
            date_normalized = date_normalized[None]
        assert date_normalized.ndim == 1, f'Input dates should be 0 or 1 dimensional, got {date_normalized.ndim}D'
        t1 = abs(abs(0.5 - date_normalized) - 0.5) + 0.05
        t2 = abs(abs(0.25 - date_normalized) - 0.5) + 0.05
        # альтернативное преобразование через синус
        # s1 = (np.cos(day * 2 * np.pi + np.pi) + 1) / 4 + 0.05
        # s2 = (np.sin(day * 2 * np.pi) + 1) / 4 + 0.05
        time_encoded = (periodic_law(frequency * np.einsum('i,jk->ijk', t1, lon))
                        + periodic_law(frequency * np.einsum('i,jk->ijk', t2, lat))) / 2
        return time_encoded
    def load_scatters(self, filenames):
        border_ts = [self.get_date_by_datefile_id(0, np_like=True), self.get_date_by_datefile_id(-1, np_like=True)]
        with xr.open_dataset(filenames[0]) as ds1, xr.open_dataset(filenames[1]) as ds2:
            xx, yy = np.meshgrid(ds1['longitude'][:].data,
                                 np.flip(ds1['latitude'][:].data))
            self.metadata['scat_xx'] = xx
            self.metadata['scat_yy'] = yy
            scat_coords = np.stack([xx.flatten(), yy.flatten()]).T
            dates = ds1['time'][:]
            slices = np.where((dates == border_ts[0]) | (dates == border_ts[1]))[0]
            print(slices[0], slices[1] + 1, 'scatter slices')
            scatter_data = []
            for ds in [ds1, ds2]:
                mask = (np.isnan(ds['eastward_wind'][slices[0]:slices[1] + 1].data) * -1 + 1) * \
                       (np.isnan(ds['northward_wind'][slices[0]:slices[1] + 1].data) * -1 + 1) * \
                       (np.isnan(ds['measurement_time'][slices[0]:slices[1] + 1].data) * -1 + 1)

                neighbour_mask = create_mask_by_nearest_to_nans(wind=ds['eastward_wind'][slices[0]:slices[1] + 1].data,
                                                                coords=scat_coords, fill_value=0)
                mask = mask * neighbour_mask
                scatter_data.append(np.flip(np.stack([ds['eastward_wind'][slices[0]:slices[1] + 1].data,
                                                      ds['northward_wind'][slices[0]:slices[1] + 1].data,
                                                      mask,
                                                      ds['measurement_time'][slices[0]:slices[1] + 1].data * 1e-9], 1),
                                            -2).copy())
            self.scatter_data = np.stack(scatter_data, 1)
            self.scatter_data = np.nan_to_num(self.scatter_data, copy=False, nan=0.)
            print(self.scatter_data.shape, 'scatter data shape')

    def load_stations(self, station_files):
        names = []
        coords = []
        stations = []
        for file in station_files:
            with open(file, 'rb') as f:
                measurements = pickle.load(f)
                names.append(measurements['Name'])
                coords.append(measurements['Coords'])
                stations.append(measurements['Station'])
        stations = np.swapaxes(np.array(stations), 0, 1)  # size (40369, 46, 4)
        border_ts = [self.get_date_by_datefile_id(0).timestamp(), self.get_date_by_datefile_id(-1).timestamp()]
        dates = stations[:, 0, 0]
        slices = np.where((dates == border_ts[0]) | (dates == border_ts[1]))[0]
        self.stations = stations[slices[0]:slices[1] + 24]
        # self.stations[np.isnan(self.stations)] = 0  # todo заменить маской
        print(self.stations.shape, pdl.from_timestamp(self.stations[0, 0, 0]),
              pdl.from_timestamp(self.stations[-1, 0, 0]), 'stations dates range')
        self.metadata['start_date'] = self.stations[0, 0, 0]
        self.metadata['end_date'] = self.stations[-1, 0, 0]
        coords = np.array(coords)
        coords[:, [0, 1]] = coords[:, [1, 0]]
        self.metadata['coords'] = coords

    def load_grid_metadata(self):
        self.metadata['wrf_xx'] = np.load(os.path.join(cfg.data.base_folder, 'metadata', 'wrf_xx.npy'))
        self.metadata['wrf_yy'] = np.load(os.path.join(cfg.data.base_folder, 'metadata', 'wrf_yy.npy'))
        self.metadata['era_xx'] = np.load(os.path.join(cfg.data.base_folder, 'metadata', 'era_xx.npy'))
        self.metadata['era_yy'] = np.load(os.path.join(cfg.data.base_folder, 'metadata', 'era_yy.npy'))

    def load_grids(self, era_grid, wrf_grid):
        pass

    @staticmethod
    def load_file_vars(filename, variables):
        npy = []
        with netCDF4.Dataset(filename, 'r') as ncf:
            for i, variable in enumerate(variables):
                var = wrf.getvar(ncf, variable, wrf.ALL_TIMES, meta=False)
                if var.ndim == 3:
                    var = np.expand_dims(var, 0)
                npy.append(var)
        npy = np.concatenate(npy, 0)
        return np.transpose(npy, (1, 0, 2, 3))

    def __len__(self):
        l = len(self.wrf_files) * self.file_len - self.seq_len + 1
        return l if l > 0 else 0

    def get_data_by_id(self, i, file_attr, var_attr):
        needed_len = self.seq_len
        path_i, item_i = self.get_path_id(i)
        npys = []
        while needed_len > 0:
            file_vars = self.load_file_vars(getattr(self, file_attr)[path_i], getattr(self, var_attr))
            part = file_vars[item_i:item_i + needed_len]
            item_i = 0
            path_i += 1
            npys.append(part)
            needed_len -= len(part)
        return np.concatenate(npys)

    def get_path_id(self, i):
        path_id, item_id = i // self.file_len, i % self.file_len
        return path_id, item_id

    def get_station(self, i):
        out = self.stations[i:i + self.seq_len]
        return out

    def get_scatter(self, i):
        out = self.scatter_data[i // 24]
        return out

    def __getitem__(self, i):
        data = self.get_data_by_id(i, 'wrf_files', 'wrf_variables')
        data = np.flip(data, 2).copy()
        encodings = [data]
        if self.use_landmask:
            encodings.append(np.broadcast_to(self.landmask, [self.seq_len, 1, *self.landmask.shape]).copy())
        if self.use_time_encoding:
            day_encoded, hour_encoded = self.get_day_hour_encoding(i)
            encodings.extend([np.expand_dims(day_encoded, 1), np.expand_dims(hour_encoded, 1)])
        data = np.concatenate(encodings, axis=1)

        target = self.get_data_by_id(i, 'era_files', 'era_variables')

        stations, scatter = None, None
        if self.stations is not None:
            stations = self.get_station(i)
        if self.scatter_data is not None:
            scatter = self.get_scatter(i)
        return data, target, stations, scatter, i


def custom_collate(batch):
    elem = batch[0]
    if isinstance(elem, type(None)):
        return None
    elif isinstance(elem, tuple):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.
        return [custom_collate(samples) for samples in transposed]  # Backwards compatibility.
    else:
        return default_collate(batch)


class TestSampler:
    def __init__(self, data_len, seq_len):
        self.data_len = data_len
        self.seq_len = seq_len

    def __len__(self):
        return self.data_len

    def __iter__(self):
        return iter(range(0, len(self), self.seq_len))


def find_files(directory, pattern):
    import os, fnmatch
    flist = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                filename = filename.replace('\\', '/')
                flist.append(filename)
    return flist


if __name__ == "__main__":
    import sys

    sys.path.insert(0, '/app/Precipitation-Nowcasting-master')
    from correction.data.train_test_split import split_train_val_test

    wrf_folder = '/app/wrf_test_dataset/'
    era_folder = '/app/era_test/'
    train_files, val_files, test_files = split_train_val_test(wrf_folder, era_folder, 0.7, 0.1, 0.2)

    train_dataset = WRFDataset(train_files[0], train_files[1])
    dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=4)
    from time import time

    st = time()
    for data1, target1 in train_dataset:
        print(data1.shape, target1.shape)
    print(time() - st)

    # print(dts.__getitem__(43)[0].shape, dts.__getitem__(43)[1].shape,)

    # def get_data_by_id_reserve(self, i, file_attr, var_attr):
    #     path_i, item_i = self.get_path_id(i)
    #     path_j, item_j = self.get_path_id(i + self.seq_len)
    #     npys = []
    #     for idx in range(path_i, path_j + 1):
    #         file_vars = self.load_file_vars(getattr(self, file_attr)[idx], getattr(self, var_attr))
    #         # file_vars = self.test_load(getattr(self, file_attr)[idx], getattr(self, var_attr))
    #         npys.append(file_vars)
    #     npys[0] = npys[0][item_i:]
    #     npys[-1] = npys[-1][:(item_j - self.file_len)]
    #
    #     return np.concatenate(npys)
