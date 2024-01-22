import numpy as np
# import pandas as pd
# import torch
from torch.utils.data import DataLoader, Dataset
from correction.config.config import cfg
# import os
import netCDF4
import wrf
import pickle
import os
import pendulum as pdl


class WRFDataset(Dataset):
    def __init__(self, wrf_files, era_files, station_files=None,
                 wrf_variables=None, era_variables=None, seq_len=4,
                 use_spatiotemporal_encoding=False, use_time_encoding=False, file_wise=False):
        super().__init__()

        print(len(wrf_files), 'wrf_files')
        print(len(era_files), 'era_files')
        wrf_files.sort()
        era_files.sort()
        self.wrf_files = wrf_files
        self.era_files = era_files

        if era_variables is None:
            era_variables = ['u10', 'v10', 't2m']
        self.era_variables = era_variables
        if wrf_variables is None:
            wrf_variables = ['uvmet10', 'T2']
        if use_spatiotemporal_encoding:
            wrf_variables.extend(['LANDMASK', 'XLAT', 'XLONG'])
            self.create_wrf_meshgrid()
        self.wrf_variables = wrf_variables

        self.metadata = {}
        self.stations = None
        if station_files is not None:
            self.load_stations(station_files)
        self.load_grid_metadata()

        self.seq_len = seq_len
        self.file_len = 24

        self.use_time_encoding = use_time_encoding
        self.use_spatiotemporal_encoding = use_spatiotemporal_encoding
        self.file_wise = file_wise

    def create_wrf_meshgrid(self):
        x, y = np.arange(0, 280, 1), np.arange(0, 210, 1)
        self.wrf_xy = np.meshgrid(x, y)

    def get_date_by_datefile_id(self, datefile_id):
        return pdl.parse(self.wrf_files[datefile_id].split('_')[-2])

    def get_time_encoding(self, i, frequency=1):
        datefile_id, hour = self.get_path_id(i)
        date = self.get_date_by_datefile_id(datefile_id)
        day = [date.add(hours=i).day_of_year for i in range(self.seq_len)]
        day = np.array(day) / 365
        hour = np.mod(np.array(range(hour, hour + self.seq_len)), 24) / 24
        d1 = abs(abs(0.5 - day) - 0.5) + 0.05
        d2 = abs(abs(0.25 - day) - 0.5) + 0.05
        h1 = abs(abs(0.5 - hour) - 0.5) + 0.05
        h2 = abs(abs(0.25 - hour) - 0.5) + 0.05
        np.einsum('i,jk->ijk', d1, self.wrf_xy[0])
        day_encoded = (np.sin(frequency * np.einsum('i,jk->ijk', d1, self.wrf_xy[0]))
                       + np.sin(frequency * np.einsum('i,jk->ijk', d2, self.wrf_xy[1]))) / 2
        hour_encoded = (np.cos(frequency * np.einsum('i,jk->ijk', h1, self.wrf_xy[0]))
                        + np.cos(frequency * np.einsum('i,jk->ijk', h2, self.wrf_xy[1]))) / 2
        return day_encoded, hour_encoded

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
        self.stations = stations[slices[0]:slices[1]+24]
        print(self.stations.shape, pdl.from_timestamp(self.stations[0, 0, 0]),
              pdl.from_timestamp(self.stations[-1, 0, 0]), 'stations dates range')
        coords = np.array(coords)
        coords[:, [0, 1]] = coords[:, [1, 0]]
        self.metadata['coords'] = coords

    def load_grid_metadata(self):
        self.metadata['wrf_xx'] = np.load(os.path.join(cfg.GLOBAL.BASE_DIR, 'metadata', 'wrf_xx.npy'))
        self.metadata['wrf_yy'] = np.load(os.path.join(cfg.GLOBAL.BASE_DIR, 'metadata', 'wrf_yy.npy'))
        self.metadata['era_xx'] = np.load(os.path.join(cfg.GLOBAL.BASE_DIR, 'metadata', 'era_xx.npy'))
        self.metadata['era_yy'] = np.load(os.path.join(cfg.GLOBAL.BASE_DIR, 'metadata', 'era_yy.npy'))

    def load_grids(self, era_grid, wrf_grid):
        pass

    def load_file_vars(self, filename, variables):
        npy = []
        with netCDF4.Dataset(filename, 'r') as ncf:
            for i, variable in enumerate(variables):
                var = wrf.getvar(ncf, variable, wrf.ALL_TIMES, meta=False)
                if len(var.shape) == 3:
                    var = np.expand_dims(var, 0)
                npy.append(var)
        npy = np.concatenate(npy, 0)
        return np.transpose(npy, (1, 0, 2, 3))

    def __len__(self):
        if self.file_wise:
            return len(self.wrf_files)
        l = len(self.wrf_files) * self.file_len - self.seq_len + 1
        return l if l > 0 else 0

    def get_data_by_id(self, i, file_attr, var_attr):
        if self.file_wise:
            return self.load_file_vars(getattr(self, file_attr)[i], getattr(self, var_attr))

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

    def __getitem__(self, i):
        data = self.get_data_by_id(i, 'wrf_files', 'wrf_variables')
        data = np.flip(data, 2).copy()
        if self.use_time_encoding:
            day_encoded, hour_encoded = self.get_time_encoding(i)
            data = np.concatenate([data, np.expand_dims(day_encoded, 1), np.expand_dims(hour_encoded, 1)], axis=1)

        target = self.get_data_by_id(i, 'era_files', 'era_variables')

        if self.stations is not None:
            stations = self.get_station(i)

            # print(self.get_path_id(i), self.wrf_files[self.get_path_id(i)[0]],
            #       self.era_files[self.get_path_id(i)[0]], pdl.from_timestamp(stations[0, 0, 0]),
            #       'sample: path id, wrf name, era name, station time')
            return data, target, stations, i
        else:
            return data, target


class WRFNPDataset(Dataset):
    def __init__(self, wrf_files, era_files, seq_len=4, station_files=None):
        super().__init__()
        print(len(wrf_files), 'wrf_files')
        if era_files is not None:
            print(len(era_files), 'era_files')
        wrf_files.sort()
        if era_files is not None:
            era_files.sort()
        self.wrf_files = wrf_files
        if era_files is not None:
            self.era_files = era_files
        names = []
        coords = []
        stations = []
        if station_files is not None:
            for file in station_files:
                with open(file, 'rb') as f:
                    measurements = pickle.load(f)
                    names.append(measurements['Name'])
                    coords.append(measurements['Coords'])
                    stations.append(measurements['Station'])
        self.stations = np.swapaxes(np.array(stations), 0, 1)
        self.seq_len = seq_len
        self.file_len = 24

    def __len__(self):
        l = len(self.wrf_files) * self.file_len - self.seq_len + 1
        return l if l > 0 else 0

    def get_data_by_id(self, i, file_attr, var_attr):
        needed_len = self.seq_len
        path_i, item_i = self.get_path_id(i)
        npys = []
        while needed_len > 0:
            file_vars = np.load(getattr(self, file_attr)[path_i])
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
        print(self.stations.shape)
        out = self.stations[i:i + self.seq_len]
        return out

    def __getitem__(self, i):
        station = self.get_station(i)
        data = self.get_data_by_id(i, 'wrf_files', 'wrf_variables')
        data = np.flip(data, 2).copy()
        if hasattr(self, 'era_files'):
            target = self.get_data_by_id(i, 'era_files', 'era_variables')
            return data, target, station
        else:
            return data


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
