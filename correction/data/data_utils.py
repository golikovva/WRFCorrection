import os
from abc import abstractmethod
from pathlib import Path
from datetime import datetime
import random
import pickle
import torch
import numpy as np
import pandas as pd
import netCDF4
import wrf
import xarray as xr
from typing import Callable, List, Dict, Tuple, Any
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import PackedSequence
from addict import Dict


def atleast_nd(arr, n):
    """
    inplace operator to expand array dims up to n dimensional
    """
    arr.shape = (1,) * (n - arr.ndim) + arr.shape
    return arr

def get_hours_in_period(date, period):
    return ((date.astype(f'datetime64[{period}]') + np.timedelta64(1, period)).astype(f'datetime64[h]') - date.astype(f'datetime64[{period}]')).astype(int)

def lon_to_180_range(lon):
    return (lon + 180) % 360 - 180

def arange_month(date):
    month = date.astype('datetime64[M]')
    res = np.arange(
        month.astype('datetime64[D]'),
        (month + np.timedelta64(1, 'M')).astype('datetime64[D]'),
        np.timedelta64(1, 'D')
    )
    return res


class NCs2sDataset(Dataset):
    def __init__(self, data_folder, data_variables=None, seq_len=4, time_resolution_h=1, add_coords=False, add_time_encoding=False):
        super().__init__()
        self.path = Path(data_folder)
        self.dates_dict = self._create_dates_dict()
        self.data_variables = data_variables

        self.constant_vars = {}
        self.src_grid = self._create_grid()
        # self.src_grid['longitude'] = lon_to_180_range(self.src_grid['longitude'])
        self.grid = Dict(self.src_grid)
        self.seq_len = seq_len
        self.file_len = 24
        self.time_res_h = time_resolution_h
        self.add_coords = add_coords
        self.add_time_encoding = add_time_encoding

    def _create_dates_dict(self):
        result = {}
        files = sorted(self.path.glob(self._files_template))
        print('found', len(files), 'files')
        for file in files:
            try:
                date = self._parse_date(file).astype(f'datetime64[{self._file_len}]')
            except ValueError:
                print(f'skipping {file}')
                continue
            if date in result:
                result[date].append(file)
            else:
                result[date] = [file]
        print(f'parsed {len(result)} dates')
        return result

    @abstractmethod
    def _create_grid(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _parse_date(file):
        raise NotImplementedError

    @property
    @abstractmethod
    def _files_template(self):
        raise NotImplementedError

    @staticmethod
    def load_file_vars(filename, variables, times=None):
        times = wrf.ALL_TIMES if times is None else times
        npy = []
        with netCDF4.Dataset(filename, 'r') as ncf:
            for i, variable in enumerate(variables):
                var = wrf.getvar(ncf, variable, times, meta=False, squeeze=False)
                var = atleast_nd(var, 5)[:, 0]
                npy.append(var)
        npy = np.concatenate(npy, 0)
        return np.transpose(npy, (1, 0, 2, 3))

    def __len__(self):
        l = len(self.dates_dict) * self.file_len - self.seq_len + 1
        return l if l > 0 else 0

    def get_data_by_id(self, date, length=None):
        needed_len = self.seq_len if length is None else length
        npys = []
        hour = (date.astype('datetime64[h]') - date.astype(f'datetime64[{self._file_len}]')).astype(int)
        file = date.astype(f'datetime64[{self._file_len}]')
        while needed_len > 0:
            times = np.arange(0, get_hours_in_period(date, self._file_len))[hour:hour + needed_len*self.time_res_h:self.time_res_h]
            try:
                file_vars = self.load_file_vars(str(self.dates_dict[file][0]), self.data_variables, times)
            except IndexError:
                print(date)
                print("No data for this dates")
                return None
            hour = 0
            file = file + np.timedelta64(1, self._file_len)
            npys.append(file_vars)
            needed_len -= len(file_vars)
        return np.concatenate(npys) 

    def __getitem__(self, date, length=None, add_coords=None, add_time_encoding=None):
        data = [self.get_data_by_id(date, length)]
        if data[0] is None:
            return None
        add_coords = self.add_coords if add_coords is None else add_coords
        add_time_encoding = self.add_time_encoding if add_time_encoding is None else add_time_encoding
        if add_coords:
            data.append(np.broadcast_to(np.stack([self.src_grid['latitude'], self.src_grid['longitude']]), [self.seq_len, 2, *data[0].shape[-2:]]))
        if self.add_time_encoding:
            day_encoded, hour_encoded = self.get_day_hour_encoding(date)
            data.extend([np.expand_dims(day_encoded, 1), np.expand_dims(hour_encoded, 1)])
        data = np.concatenate(data, axis=1)
        return data
    
    def get_day_hour_encoding(self, date, frequency=1):
        date_seq = np.arange(date, date + np.timedelta64(self.seq_len*self.time_res_h, 'h'), np.timedelta64(self.time_res_h, 'h'))
        day = (date_seq.astype('datetime64[D]') - date_seq.astype('datetime64[Y]')).astype(int) / 365
        hour = (date_seq.astype('datetime64[h]') - date_seq.astype('datetime64[D]')).astype(int) / 24
        d1 = abs(abs(0.5 - day) - 0.5) + 0.05
        d2 = abs(abs(0.25 - day) - 0.5) + 0.05
        h1 = abs(abs(0.5 - hour) - 0.5) + 0.05
        h2 = abs(abs(0.25 - hour) - 0.5) + 0.05
        x, y = np.arange(0, self.src_grid['latitude'].shape[1], 1), np.arange(0, self.src_grid['latitude'].shape[0], 1)
        xxyy = np.meshgrid(x, y)
        day_encoded = (np.sin(frequency * np.einsum('i,jk->ijk', d1, xxyy[0]))
                       + np.sin(frequency * np.einsum('i,jk->ijk', d2, xxyy[1]))) / 2
        hour_encoded = (np.cos(frequency * np.einsum('i,jk->ijk', h1, xxyy[0]))
                        + np.cos(frequency * np.einsum('i,jk->ijk', h2, xxyy[1]))) / 2
        # if debug:
        #     return day_encoded, hour_encoded, day, hour, d1, d2,
        return day_encoded, hour_encoded


class WRFs2sDataset(NCs2sDataset):
    @staticmethod
    @abstractmethod
    def _parse_date(file):
        date_part = file.name.split('_')[2]
        date = np.datetime64(date_part)
        return date

    @property
    def _files_template(self):
        return '*wrfout*'

    def _create_grid(self):
        grid_path = sorted(self.path.glob(self._files_template))[0]
        print(f'loading grid from {grid_path}')
        with xr.open_dataset(grid_path, cache=False) as ds:
            lon = ds.coords['XLONG'].values[0]
            lat = ds.coords['XLAT'].values[0]
            grid = {'longitude': lon, 'latitude': lat}
        return grid
    
    @property
    def _file_len(self):
        return 'D'

class ERAs2sDataset(NCs2sDataset):
    @staticmethod
    @abstractmethod
    def _parse_date(file):
        date_part = file.name.split('_')[-1]
        date = np.datetime64(date_part)
        return date

    @property
    def _files_template(self):
        return '*era*'

    def _create_grid(self):
        grid_path = sorted(self.path.glob(self._files_template))[0]
        print(f'loading grid from {grid_path}')
        with xr.open_dataset(grid_path) as ds:
            lat_vec = ds.coords['latitude'].values
            lon_vec = ds.coords['longitude'].values

            lat = np.tile(lat_vec, (lon_vec.size, 1)).T
            lon = np.tile(lon_vec, (lat_vec.size, 1))

            grid = {'longitude': lon, 'latitude': lat}
        return grid
    
    @property
    def _file_len(self):
        return 'D'
        
class ERAMonthlyDataset(ERAs2sDataset):
    @staticmethod
    def _parse_date(file):
        """
        Extract year and month from monthly file name (e.g., ERA5_2023-10.nc).
        """
        date_part = file.name.split('_')[-1].split('.')[0]  # Extract "2023-10"
        date = np.datetime64(date_part + '-01')  # Convert to first day of the month
        return date
    
    @property
    def _file_len(self):
        return 'M'
    
    # @property
    # def _files_template(self):
    #     return 'era*.nc'


    # def _create_grid(self):
    #     """
    #     Create grid from the first file.
    #     """
    #     grid_path = sorted(self.path.glob(self._files_template))[0]
    #     print(f'Loading grid from {grid_path}')
    #     with xr.open_dataset(grid_path) as ds:
    #         lat_vec = ds.coords['latitude'].values
    #         lon_vec = ds.coords['longitude'].values

    #         lat = np.tile(lat_vec, (lon_vec.size, 1)).T
    #         lon = np.tile(lon_vec, (lat_vec.size, 1))

    #         grid = {'longitude': lon, 'latitude': lat}
    #     return grid
        
# class ERAMonthlyDataset_nh(ERAMonthlyDataset):
#     def __init__(self, data_folder, data_variables=None, seq_len=4, time_res_h=6):
#         super().__init__(data_folder, data_variables, seq_len)
#         self.time_res_h = time_res_h
        
#     def get_data_by_id(self, date, length=None):
#         needed_len = self.seq_len if length is None else length
#         npys = []
#         hour = (date.astype('datetime64[h]') - date.astype(f'datetime64[{self._file_len}]')).astype(int)
#         file = date.astype(f'datetime64[{self._file_len}]')
#         while needed_len > 0:
#             times = np.arange(0, get_hours_in_period(date, self._file_len))[hour:hour + needed_len*self.time_res_h:self.time_res_h]
#             try:
#                 file_vars = self.load_file_vars(str(self.dates_dict[file][0]), self.data_variables, times)
#             except IndexError:
#                 print(date)
#                 print("No data for this dates")
#                 return None
#             hour = 0
#             file = file + np.timedelta64(1, self._file_len)
#             npys.append(file_vars)
#             needed_len -= len(file_vars)
#         return np.concatenate(npys) 


class ScatterNoneDataset(Dataset):
    def __getitem__(self, index, *args, **kwargs):
        return None


class StationsNoneDataset(Dataset):
    def __getitem__(self, index, *args, **kwargs):
        return None


class StationsDataset(Dataset):
    """
    PyTorch Dataset for station-based surface weather time series.

    Each item is a block of shape (length, n_stations, n_vars),
    optionally converting wind speed & direction into u/v components.

    Args:
        stations_folder (str): Directory containing station .pkl files. All stations should have the same shape.
        data_variables (list[str], optional): List of non-wind variable names. Defaults to ['TC'].
        wind_variables (list[str], optional): (u,v) or (speed, dir) components. Defaults to ['WSPD10','WDIR10_angle_interp'].
        time_variable (str, optional): Name of the timestamp column in each DataFrame. Defaults to 'timestamp'.
        seq_len (int, optional): Default sequence length returned. Defaults to 4.
        from_direction (bool, optional): If True, randomly sample (u, v) from (speed, dir). Defaults to True.
    """
    def __init__(
        self,
        stations_folder: str,
        data_variables: list[str] = None,
        wind_variables: list[str] = None,
        time_variable: str = 'timestamp',
        seq_len: int = 4,
        wind_format: str = 'uv',   # 'uv' or 'wd'
        rng=None
    ):
        self.path = Path(stations_folder)
        self.station_files = sorted(self.path.glob(self._files_template))
        if data_variables is None:
            data_variables = ['TC']
        if wind_variables is None:
            wind_variables = ['WSPD10', 'WDIR10_angle_interp']
        assert wind_format in ('uv', 'wd'), "wind_format must be 'uv' or 'wd'"
        
        self.data_variables = wind_variables + data_variables
        self.time_variable = time_variable
        self.seq_len = seq_len
        self.wind_format = wind_format
        self.rng = rng or np.random.default_rng()

        self.stations = self.load_stations(self.station_files)
        self.dates_dict = self._create_dates_dict()
        self.src_grid = self._create_grid()

    @staticmethod
    def angle_to_sector_class(angle: np.ndarray, num_sectors: int = 16) -> np.ndarray:
        sector_size = 360.0 / num_sectors
        idx = np.floor((angle + sector_size/2) / sector_size) % num_sectors
        return idx
    
    @staticmethod
    def _sample_uv(arr: np.ndarray, ws_idx: int, ang_idx: int,
                   sector_size: float, rng=None) -> np.ndarray:
        """
        Replace wind speed & direction with u,v sampled within the sector.
        arr: shape (..., N_vars)
        ws_idx: index of speed
        ang_idx: index of angle (degrees)
        """
        if rng is None:
            rng = np.random.default_rng()
        out = arr.copy()
        speed = arr[..., ws_idx]
        mid_angle = arr[..., ang_idx]
        offset = rng.uniform(-sector_size/2, sector_size/2, size=speed.shape)
        random_angle = mid_angle + offset
        theta = np.deg2rad(random_angle)
        u = -speed * np.sin(theta)
        v = -speed * np.cos(theta)
        out[..., ws_idx] = u
        out[..., ang_idx] = v
        return out
    
    def _create_grid(self):
        print(self.coords.shape, 'coords shape inside _create_grid')
        grid = {'longitude': self.coords[:, 0], 'latitude': self.coords[:, 1]}
        return grid

    def _create_dates_dict(self):
        with open(self.station_files[0], 'rb') as f:
            meas = pickle.load(f)
        idx = pd.Series(
            np.arange(len(meas['Station'][self.time_variable])),
            index=meas['Station'][self.time_variable]
        )
        return idx

    def load_stations(self, station_files):
        names = []
        coords = []
        stations = []
        for file in station_files:
            with open(file, 'rb') as f:
                measurements = pickle.load(f)
                names.append(measurements['Name'])
                coords.append(measurements['Coords'])
                subset = measurements['Station'][self.data_variables]
                stations.append(subset.to_numpy())
        stations = np.stack(stations, axis=1)  # size (N_dates, n_stations, variables)
        self.coords = np.stack(coords, axis=0)[:, [1, 0]] # size (N_dates, n_stations, 2)
        self.names=names
        return stations

    def __getitem__(self, date_index, length=None):
        length = self.seq_len if length is None else length
        start = date_index
        end = date_index + np.timedelta64(length-1, 'h')
        ids = self.dates_dict[start:end]
        
        seq = self.stations[ids]  # shape (length, n_stations, n_vars)
        # sample u,v for wind: speed idx=0, angle idx=1
        if self.wind_format == 'uv':
            seq = self._sample_uv(arr=seq, ws_idx=0, ang_idx=1, sector_size=360/16, rng=self.rng)
        else:
            out = seq.copy()
            speed = out[..., 0]
            angle = out[..., 1]
            sector = self.angle_to_sector_class(angle, num_sectors=16)
            out[..., 0] = speed
            out[..., 1] = sector
            seq = out
        return seq
    
    def __len__(self):
        return len(self.dates_dict) - self.seq_len + 1

    @property
    def _files_template(self):
        return '*.pkl'


class ScatterDataset(Dataset):
    def __init__(self, file_paths, seq_len=24, dict_out=False):
        """
        Args:
            file_paths (list of str): List of file paths to xarray datasets.
            sequence_len (int): Length of the desired scatter time sequence in hours.
        """
        if os.path.isdir(file_paths):
            self.file_paths = sorted(Path(file_paths).glob(self._files_template))
        else:
            self.file_paths = file_paths
        self.seq_len = seq_len
        self.datasets = self._load_datasets()
        self.time_index = self._build_time_index()
        self.src_grid = self._create_grid()  # Create the grid
        self.dict_out = dict_out

    def _load_datasets(self):
        """Load all xarray datasets from the file paths."""
        datasets = []
        for file_path in self.file_paths:
            ds = xr.open_dataset(file_path)
            datasets.append(ds)
        return datasets

    def _build_time_index(self):
        """Build a unified time index across all xarray datasets."""
        time_index = []
        for ds in self.datasets:
            time_index.extend(ds.time.values)
        return np.sort(np.unique(time_index))

    def _create_grid(self):
        """
        Create a grid from the first dataset.

        Returns:
            dict: A dictionary containing the longitude and latitude grid.
        """
        # Use the first dataset to extract latitude and longitude
        ds = self.datasets[0]
        lat_vec = ds.coords['latitude'].values
        lon_vec = ds.coords['longitude'].values

        # Create a grid using numpy's tile function
        lat = np.tile(lat_vec, (lon_vec.size, 1)).T
        lon = np.tile(lon_vec, (lat_vec.size, 1))

        # Return the grid as a dictionary
        grid = {'longitude': lon, 'latitude': lat}
        return grid

    def __len__(self):
        """Return the total number of possible sequences."""
        return len(self.time_index) - self.seq_len
    
    @property
    def _files_template(self):
        return '*ascat*.nc'
    
    def __getitem__(self, idx, length=None):
        """
        Args:
            idx (int or np.datetime64): Index or timestamp of the starting time.

        Returns:
            dict: A dictionary containing the scatterometer data for the sequence.
        """
        # Convert idx to a valid time index if it's a np.datetime64 object
        length = self.seq_len if length is None else length
        start_time = idx.astype('datetime64[D]')
        end_time = (idx + np.timedelta64(length-1, 'h')).astype('datetime64[D]')
        # Initialize lists to store the data for the sequence
        measurement_time_seq = []
        northward_wind_seq = []
        eastward_wind_seq = []

        for ds in self.datasets:
            # Select the time slice within the sequence
            time_slice = ds.sel(time=slice(start_time, end_time))

            if len(time_slice.time) > 0:
                measurement_time_seq.append(time_slice.measurement_time.values)
                northward_wind_seq.append(time_slice.northward_wind.values)
                eastward_wind_seq.append(time_slice.eastward_wind.values)

        # Stack the data along the time dimension
        if measurement_time_seq:  # Check if the list is not empty
            measurement_time_seq = np.concatenate(measurement_time_seq, axis=0)
            northward_wind_seq = np.concatenate(northward_wind_seq, axis=0)
            eastward_wind_seq = np.concatenate(eastward_wind_seq, axis=0)
        else:
            # If no data is found, return empty arrays
            measurement_time_seq = np.array([])
            northward_wind_seq = np.array([])
            eastward_wind_seq = np.array([])
            
        if self.dict_out:
            # Return the data as a dictionary
            return {
                'measurement_time': measurement_time_seq.astype('datetime64[ns]').astype('datetime64[s]').astype('float64'),
                'eastward_wind': eastward_wind_seq,
                'northward_wind': northward_wind_seq,
            }
        else:
            # Return the data as a tuple
            return (
                measurement_time_seq.astype('datetime64[ns]').astype('datetime64[s]').astype('float64'),
                eastward_wind_seq,
                northward_wind_seq,
            )


class IFSs2sDataset(NCs2sDataset):
    @staticmethod
    def _parse_date(file):
        date_str = file.name.split('.')[-2]
        if len(date_str) != 8 or not date_str.isdigit():
            raise ValueError(f"Invalid date format: {file.name}")
        return np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
        
    def _create_dates_dict(self):
        """Create nested dict: {date: {var: file_path}}"""
        result = {}
        files = sorted(self.path.glob(self._files_template))
        for file in files:
            date = self._parse_date(file)
            var = self._parse_variable(file)
            if date not in result:
                result[date] = {}
            result[date][var] = file
        return result

    @staticmethod
    def _parse_variable(file):
        """Extract variable name from filename (second component after split)"""
        return file.name.split('.')[1]

    @property
    def _files_template(self):
        return '*ec.oper*.*.nc'  # Adjust based on actual file pattern

    def _create_grid(self):
        grid_path = next(self.path.glob(self._files_template))
        print(f'Loading IFS grid from {grid_path}')
        with xr.open_dataset(grid_path) as ds:
            if 'latitude' in ds.variables and 'longitude' in ds.variables:
                lat = ds['latitude'].values
                lon = ds['longitude'].values
                if len(lat.shape) == 1:  # 1D vectors
                    lon, lat = np.meshgrid(lon, lat)
                grid = {'longitude': lon, 'latitude': lat}
                self.constant_vars.update(grid)
            else:
                raise ValueError("Latitude/Longitude variables not found")
        return grid

    def get_data_by_id(self, start_date, length=None):
        seq_len = self.seq_len if length is None else length
        npys = []
        current_time = pd.Timestamp(start_date)
        for _ in range(seq_len):
            # Convert to numpy datetime64 and floor to nearest day
            current_day = current_time.to_datetime64().astype('datetime64[D]')
            
            # Calculate initial time parameters
            if current_time.hour < 6:
                file_date = current_day - np.timedelta64(1, 'D')
                init_hour = 12  # Previous day's 12Z initialization
            elif current_time.hour < 18:
                file_date = current_day
                init_hour = 0   # Same day's 00Z initialization
            else:
                file_date = current_day
                init_hour = 12  # Same day's 12Z initialization

            # Convert file_date to proper datetime64[D] for dict lookup
            file_date_d = file_date.astype('datetime64[D]')
            
            # Check date availability
            if file_date_d not in self.dates_dict:
                return None

            # Calculate forecast hour
            init_time = pd.Timestamp(file_date_d) + pd.DateOffset(hours=init_hour)
            forecast_hour = (current_time - init_time).total_seconds() // 3600

            # Validate forecast hour
            if forecast_hour not in [6, 12]:
                return None

            # Get files for all variables
            date_files = self.dates_dict[file_date_d]
            missing_vars = [var for var in self.data_variables if var not in date_files and var not in self.constant_vars] 
            if missing_vars:
                return None

            # Load and stack variables
            var_data = []
            for var in self.data_variables:
                if var in self.constant_vars:
                    var_data.append(self.constant_vars[var])
                else:
                    with xr.open_dataset(date_files[var]) as ds:
                        # Select initialization time (0 or 1) and forecast hour (0=6h, 1=12h)
                        init_idx = 0 if init_hour == 0 else 1
                        fhour_idx = 0 if forecast_hour == 6 else 1
                        var_data.append(ds[var][init_idx, fhour_idx].values)
            
            npys.append(np.stack(var_data, axis=0))  # (vars, H, W)
            current_time += pd.DateOffset(hours=6)
        
        return np.stack(npys, axis=0)  # (time, vars, H, W)


class StackDataset(Dataset):
    def __init__(self, *datasets):
        self._length = len(datasets[0])
        self.datasets = datasets

    def __getitem__(self, index):
        return tuple(dataset[index] for dataset in self.datasets)

    def __len__(self):
        return self._length
    
class StackVSDataset(StackDataset):
    def __init__(self, *datasets, max_sl=None):
        super().__init__(*datasets)
        self.max_sl = self.datasets[0].seq_len if max_sl is None else max_sl

    def __getitem__(self, index, cur_sl=None):
        cur_sl = random.randint(1, self.max_sl) if cur_sl is None else cur_sl
        return tuple(dataset.__getitem__(index, cur_sl) for dataset in self.datasets)

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.

    e.g. MNISTWithIndices = dataset_with_indices(MNIST)
         dataset = MNISTWithIndices('~/datasets/mnist')
    """

    def __getitem__(self, index):
        data = cls.__getitem__(self, index)
        return *data, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


class Sampler:
    def __init__(self, days, shuffle=False):
        self.days = days
        self.shuffle = shuffle

    def __len__(self):
        return len(self.days)

    def __iter__(self):
        ids = np.arange(len(self.days))
        if self.shuffle:
            np.random.shuffle(ids)
        for i in ids:
            yield self.days[i]


def none_consistent_collate(batch):
    elem = batch[0]
    if isinstance(elem, type(None)):
        return None
    elif isinstance(elem, np.datetime64):
        return np.array(batch)
    elif isinstance(elem, tuple):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.
        return [none_consistent_collate(samples) for samples in transposed]  # Backwards compatibility.
    else:
        return default_collate(batch)

import functools
import warnings

from typing import Callable

def ignore_warnings(category: Warning):
    def ignore_warnings_decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=category)
                return func(*args, **kwargs)
        return wrapper
    return ignore_warnings_decorator

# @ignore_warnings(category=UserWarning)
# def variable_len_collate(batch):
#     """
#     Custom collate function for handling batches with None values and variable-length sequences.

#     Args:
#         batch (list): A list of data samples.
#     elif isinstance(elem, torch.Tensor) and any(len(b) != len(elem) for b in batch):
#     Returns:
#         Collated batch data, which can be None, a list of datetime64 objects, a packed sequence of tensors,
#         or a default collated batch depending on the input batch type.
#     """
#     elem = batch[0]
#     if isinstance(elem, type(None)):
#         return None
#     elif isinstance(elem, np.datetime64):
#         return np.array(batch)
#     elif isinstance(elem, np.ndarray):
#         return variable_len_collate([torch.from_numpy(item) for item in batch])
#     elif isinstance(elem, torch.Tensor):
#         return torch.nested.to_padded_tensor(torch.nested.nested_tensor(batch), padding=0.0)
#         return pack_sequence(batch, enforce_sorted=False)
#     elif isinstance(elem, tuple):
#         # check to make sure that the elements in batch have consistent size
#         it = iter(batch)
#         elem_size = len(next(it))
#         if not all(len(elem) == elem_size for elem in it):
#             raise RuntimeError('each element in list of batch should be of equal size')
#         transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.
#         return [variable_len_collate(samples) for samples in transposed]  # Backwards compatibility.
#     else:
#         return default_collate(batch)
    
def _contains_none(x):
    """Recursively check if a sample contains None anywhere."""
    if x is None:
        return True
    if isinstance(x, (list, tuple, set)):
        return any(_contains_none(v) for v in x)
    if isinstance(x, dict):
        return any(_contains_none(v) for v in x.values())
    # We assume ndarrays/tensors here never contain Python None as payload.
    return False

@ignore_warnings(category=UserWarning)
def variable_len_collate(batch):
    """
    Custom collate function for handling batches with None values and variable-length sequences.

    Args:
        batch (list): A list of data samples.
    elif isinstance(elem, torch.Tensor) and any(len(b) != len(elem) for b in batch):
    Returns:
        Collated batch data, which can be None, a list of datetime64 objects, a packed sequence of tensors,
        or a default collated batch depending on the input batch type.
    """
    batch = [b for b in batch if not _contains_none(b)]
    if len(batch) == 0:
        return None  # training loop should skip this batch

    elem = batch[0]
    if isinstance(elem, np.datetime64):
        return np.array(batch)
    
    if isinstance(elem, np.ndarray):
        return variable_len_collate([torch.from_numpy(item) for item in batch])
    
    if isinstance(elem, torch.Tensor):
        try:
            return torch.nested.to_padded_tensor(torch.nested.nested_tensor(batch), padding=0.0)
        except RuntimeError:
            # not ragged in the leading dim; let default handle strict stacking
            return default_collate(batch)
        # return torch.nested.to_padded_tensor(torch.nested.nested_tensor(batch), padding=0.0)
        # return pack_sequence(batch, enforce_sorted=False)

    if isinstance(elem, (list, tuple)):
        # check to make sure that the elements in batch have consistent size
        # it = iter(batch)
        # elem_size = len(next(it))
        # if not all(len(elem) == elem_size for elem in it):
        #     raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.
        return [variable_len_collate(samples) for samples in transposed]  # Backwards compatibility.
    
    if isinstance(elem, dict):
        keys = elem.keys()
        return {k: variable_len_collate([d[k] for d in batch]) for k in keys}
    return default_collate(batch)

def transform_packed_sequence_multiple(
    packed: PackedSequence,
    transforms: List[Tuple[Callable[[torch.Tensor, Any], torch.Tensor], Tuple[Any, ...], Dict[str, Any]]]
) -> PackedSequence:
    """
    Последовательно применяет несколько функций преобразования к данным внутри PackedSequence.
    
    Args:
        packed (PackedSequence): Исходный PackedSequence.
        transforms (List[Tuple[Callable, Tuple, Dict]]): Список преобразований, где:
            - первый элемент: функция преобразования,
            - второй элемент: кортеж позиционных аргументов для функции,
            - третий элемент: словарь именованных аргументов для функции.
    
    Returns:
        PackedSequence: Новый PackedSequence с преобразованными данными.
    """
    data = packed.data  # Исходные данные PackedSequence

    # Применяем каждую функцию последовательно
    for transform_fn, args, kwargs in transforms:
        data = transform_fn(data, *args, **kwargs)

    # Создаем новый PackedSequence с преобразованными данными
    return PackedSequence(data, packed.batch_sizes, packed.sorted_indices, packed.unsorted_indices)



def get_novaya_zemlya_mask(fill_value=torch.nan, return_vertices=False):
    nz_vertices = np.array([
        [120, 120],
        [70, 130],
        [10, 190],
        [0, 240],
        [40, 280],
        [160, 160],
    ])
    nz_polygon_array = torch.from_numpy(create_polygon([210, 280], nz_vertices))
    nz_polygon_array = torch.where(nz_polygon_array == 0, fill_value, 1)
    if not return_vertices:
        return nz_polygon_array
    return nz_polygon_array, nz_vertices


def check(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape)  # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
    sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign


def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k - 1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1

    return base_array
