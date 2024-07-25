import sys
sys.path.insert(0, '../../')
import torch
import numpy as np
from correction.config.cfg import cfg
from torch.optim import lr_scheduler
from correction.models.loss import TurbulentMSE
import os
from correction.models.changeToERA5 import MeanToERA5
from correction.data.train_test_split import split_train_val_test, find_files
from correction.data.my_dataloader import WRFDataset, custom_collate, TestSampler
from correction.data.scaler import StandardScaler
from torch.utils.data import DataLoader
from correction.pipeline.test import test
from correction.pipeline.train import train
from correction.models.build_module import build_correction_model
from correction.data.logger import WRFLogger
from correction.helpers.interpolation import InvDistTree

cfg['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folder_name = cfg.model_type
logger = WRFLogger(cfg.data.logs_folder, folder_name)

betas = [cfg.betas.beta1, cfg.betas.beta2, [cfg.betas.beta3_w10, cfg.betas.beta3_t2], cfg.betas.beta4]
print(betas, 'betas')

print('Splitting train val test...')
train_files, val_files, test_files = split_train_val_test(cfg.data.wrf_folder, cfg.data.era_folder, 0.7, 0.1, 0.2)
station_files = find_files(cfg.data.stations_folder, '*.pkl')
scatter_files = find_files(cfg.data.scatter_folder, '*')
print('Split completed!')

if 'train' in cfg.run_config.run_mode:
    train_dataset = WRFDataset(train_files[0], train_files[1], wrf_variables=cfg.data.wrf_variables,
                               station_files=station_files, scatter_files=scatter_files,
                               use_spatial_encoding=cfg.run_config.use_spatial_encoding,
                               use_time_encoding=cfg.run_config.use_time_encoding,
                               use_landmask=cfg.run_config.use_landmask)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.run_config.batch_size, shuffle=True,
                                  num_workers=cfg.run_config.num_workers,
                                  collate_fn=custom_collate, pin_memory=True)

    valid_dataset = WRFDataset(val_files[0], val_files[1], wrf_variables=cfg.data.wrf_variables,
                               station_files=station_files, scatter_files=scatter_files,
                               use_spatial_encoding=cfg.run_config.use_spatial_encoding,
                               use_time_encoding=cfg.run_config.use_time_encoding,
                               use_landmask=cfg.run_config.use_landmask)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.run_config.batch_size, shuffle=False,
                                  num_workers=cfg.run_config.num_workers,
                                  collate_fn=custom_collate)

test_dataset = WRFDataset(test_files[0], test_files[1], wrf_variables=cfg.data.wrf_variables,
                          station_files=station_files, scatter_files=scatter_files,
                          use_spatial_encoding=cfg.run_config.use_spatial_encoding,
                          use_time_encoding=cfg.run_config.use_time_encoding,
                          use_landmask=cfg.run_config.use_landmask)
test_sampler = TestSampler(len(test_dataset), 4)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg.run_config.num_workers,
                             collate_fn=custom_collate, sampler=test_sampler)


means_dict = torch.load(cfg.data.wrf_mean_path)
stds_dict = torch.load(cfg.data.wrf_std_path)
time_keys = ['day', 'hour'] if cfg.run_config.use_spatial_encoding else []
landmask_key = ['LANDMASK'] if cfg.run_config.use_landmask else []
wrf_keys = ['u10', 'v10', 'T2'] + test_dataset.wrf_variables[2:] + landmask_key + time_keys
era_keys = ['u10', 'v10', 'T2']
print(wrf_keys, '- wrf channels to transform')
print(era_keys, '- era channels to transform')

era_scaler = StandardScaler()
wrf_scaler = StandardScaler()
era_scaler.apply_scaler_channel_params(torch.tensor([means_dict[x] for x in era_keys]).float().to(cfg.device),
                                       torch.tensor([stds_dict[x] for x in era_keys]).float().to(cfg.device))
wrf_scaler.apply_scaler_channel_params(torch.tensor([means_dict[x] for x in wrf_keys]).float().to(cfg.device),
                                       torch.tensor([stds_dict[x] for x in wrf_keys]).float().to(cfg.device))
print(wrf_scaler.means, wrf_scaler.stddevs)

metadata = test_dataset.metadata

era_coords = np.stack([metadata['era_xx'].flatten(), metadata['era_yy'].flatten()]).T
wrf_coords = np.stack([metadata['wrf_xx'].flatten(), metadata['wrf_yy'].flatten()]).T
scat_coords = np.stack([metadata['scat_xx'].flatten(), metadata['scat_yy'].flatten()]).T if scatter_files else None
meaner = MeanToERA5(os.path.join(cfg.data.logs_folder, 'wrferaMapping.npy'),
                    era_coords=era_coords, wrf_coords=wrf_coords,
                    weighted=cfg.run_config.weighted_meaner)\
    .to(cfg.device)
stations_interpolator = InvDistTree(x=wrf_coords, q=metadata['coords']) if station_files else None
scat_interpolator = InvDistTree(x=wrf_coords, q=scat_coords) if scatter_files else None
criterion = TurbulentMSE(meaner, betas, stations_interpolator, scat_interpolator, logger=logger,
                         kernel_type=cfg.loss_config.loss_kernel, k=cfg.loss_config.k, device=cfg.device)

model = build_correction_model(cfg)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=1e-5)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.2)


best_epoch = cfg.test_config.best_epoch_id
if 'train' in cfg.run_config.run_mode:
    print(f"Started training the model: run no {logger.experiment_number}")
    best_epoch, encoder_forecaster = train(train_dataloader, valid_dataloader, model, optimizer, wrf_scaler,
                                           era_scaler, criterion, scheduler, logger, cfg)

print(f"Started testing the model: run no {logger.experiment_number}")
state_dict = torch.load(os.path.join(logger.model_save_dir, f'model_{best_epoch}.pth'))
model.load_state_dict(state_dict)
save_dir = logger.save_dir
results = test(model, criterion, wrf_scaler, era_scaler, test_dataloader, logger, cfg)
print(results)

