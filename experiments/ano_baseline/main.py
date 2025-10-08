import sys
import os

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np


sys.path.insert(0, '../../')
from correction.data.train_test_split import split_dates, split_dates_by_dates
from correction.data.my_dataloader import none_consistent_numpy_collate
from correction.data.data_utils import StackDataset, Sampler, WRFs2sDataset, ERAs2sDataset, dataset_with_indices
from correction.models.ano_corr_accumulator import CorrAccumulator
from correction.helpers.interpolation import InvDistTree
from correction.config.cfg import cfg
# cfg['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean_period = 'day'

wrf_variables = ['uvmet10', 'T2']
era_variables = ['u10', 'v10', 't2m']
wrf_dataset = WRFs2sDataset(cfg.data.wrf_folder, wrf_variables, seq_len=24)
era_dataset = ERAs2sDataset(cfg.data.era_folder, era_variables, seq_len=24)
dataset = dataset_with_indices(StackDataset)(wrf_dataset, era_dataset)

start_date, end_date = np.datetime64(cfg.data.start_date), np.datetime64(cfg.data.end_date)
print(start_date, end_date)

train_days, val_days, test_days = split_dates(start_date, end_date, .7, .1, .2, time_step='D')
print(len(train_days))
train_days = train_days[~(train_days == np.datetime64('2020-12-31'))]
print(len(train_days))

train_sampler = Sampler(train_days, shuffle=False)
train_loader = DataLoader(dataset, batch_size=10, num_workers=10, sampler=train_sampler,
                          collate_fn=none_consistent_numpy_collate)

wrf_grid, era_grid = wrf_dataset.src_grid, era_dataset.src_grid
era_coords = np.stack([era_grid['longitude'].flatten(), era_grid['latitude'].flatten()]).T
wrf_coords = np.stack([wrf_grid['longitude'].flatten(), wrf_grid['latitude'].flatten()]).T
interpolator = InvDistTree(x=era_coords, q=wrf_coords, n_near=4)


model = CorrAccumulator(start_date, mean_period)


for data, target, i in (pbar := tqdm(train_loader)):
    print(i)
    i = np.squeeze(i)
    print(i.shape, data.shape, target.shape)
    target = interpolator(torch.from_numpy(target).flatten(-2, -1)).reshape(data.shape).numpy()
    corr = target - data
    print(i.shape, corr.shape)
    model.accumulate_corr(corr.mean(0), i)
    print((i - i.astype('datetime64[Y]')).astype(int))
    if (i - i.astype('datetime64[Y]')).astype(int)[0] % 50 == 0:
        print(corr.max(axis=(1, 2, 3)))
        print(model.period_counts)
        print(model.period_means.mean(axis=(1, 2, 3)))

os.makedirs(os.path.join(cfg.data.logs_folder, 'ano_baseline'), exist_ok=True)
model.save_correction_fields(os.path.join(cfg.data.logs_folder, 'ano_baseline',
                                          f'{mean_period}_correction_fields.npy'))
