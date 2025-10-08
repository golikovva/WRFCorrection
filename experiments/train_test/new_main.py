import os
import sys
sys.path.insert(0, '../../')
import numpy as np

import torch
from torch.optim import lr_scheduler

from correction.models.loss import TurbulentMSE, UpdatedTurbulentMSE
from correction.models.changeToERA5 import MeanToERA5, ClusterMapper
from correction.data.train_test_split import split_train_val_test, find_files, split_dates_by_dates, split_dates
from correction.data.my_dataloader import WRFDataset, none_consistent_collate, TestSampler
from correction.data.data_utils import WRFs2sDataset, ERAs2sDataset, ScatterDataset, StationsDataset, StackDataset, \
    dataset_with_indices, Sampler, ScatterNoneDataset, StationsNoneDataset, StackVSDataset, variable_len_collate
from correction.data.scaler import StandardScaler
from torch.utils.data import DataLoader
from correction.pipeline.test import test
from correction.pipeline.train import train
from correction.models.build_module import build_correction_model
from correction.data.logger import WRFLogger
from correction.helpers.interpolation import InvDistTree



def main(cfg):
    folder_name = 'samples/bunet_raw_misc_3_best_rmse'#cfg.model_type
    logger = WRFLogger(cfg, cfg.data.logs_folder, folder_name)
    print(logger.model_save_dir, 'model save dir')
    logger.model_save_dir = os.path.join('/home/logs/samples', 'bunet_raw_misc_3_best_rmse', 'models')
    if not cfg.run_config.run_mode == 'test':
        print('Saving config')
        cfg.save_config(os.path.join(logger.save_dir, "config_used.yaml"))

    cfg.device = torch.device(cfg.device)
    print(f"Running on {cfg.device} device")
    betas = [cfg.betas.beta1, cfg.betas.beta2, cfg.betas.beta3_t2, cfg.betas.beta4]
    print(betas, 'betas')

    print('Splitting train val test...')
    max_sl = cfg.s2s.sequence_len
    wrf_dataset = WRFs2sDataset(cfg.data.wrf_folder, cfg.data.wrf_variables, seq_len=max_sl, 
                                add_coords=cfg.run_config.use_spatial_encoding,
                                add_time_encoding=cfg.run_config.use_time_encoding)
    era_dataset = ERAs2sDataset(cfg.data.era_folder, cfg.data.era_variables, seq_len=max_sl)
    # st_ds, sc_ds = StationsNoneDataset(), ScatterNoneDataset()
    st_ds = StationsDataset(cfg.data.stations_folder,seq_len=max_sl)
    sc_ds = ScatterDataset(cfg.data.scatter_folder, seq_len=max_sl)
    dataset = dataset_with_indices(StackDataset)(wrf_dataset, era_dataset, st_ds, sc_ds)

    start_date, end_date = np.datetime64(cfg.data.start_date), np.datetime64(cfg.data.end_date)
    train_days, val_days, test_days = split_dates(start_date, end_date, 0.7, 0.1, 0.2)
    train_days = train_days[
        ~((train_days >= np.datetime64('2020-12-31T00') - np.timedelta64(max_sl, 'h')) & (train_days <= np.datetime64('2020-12-31T23')))]
    print('Split completed!')

    train_sampler = Sampler(train_days, shuffle=True)
    val_sampler = Sampler(val_days, shuffle=False)
    test_sampler = Sampler(test_days, shuffle=False)
    collate_fn = variable_len_collate if cfg.run_config.variable_sequence_length else variable_len_collate

    train_dataloader = DataLoader(dataset, batch_size=cfg.run_config.batch_size, num_workers=cfg.run_config.num_workers,
                                sampler=train_sampler, collate_fn=collate_fn, pin_memory=True)
    valid_dataloader = DataLoader(dataset, batch_size=cfg.run_config.batch_size, num_workers=cfg.run_config.num_workers,
                                sampler=val_sampler, collate_fn=collate_fn, pin_memory=True)
    test_dataloader = DataLoader(dataset, batch_size=cfg.run_config.batch_size, num_workers=cfg.run_config.num_workers,
                                 sampler=test_sampler, collate_fn=collate_fn, pin_memory=True)

    means_dict = torch.load(cfg.data.wrf_mean_path)
    stds_dict = torch.load(cfg.data.wrf_std_path)
    time_keys = ['day', 'hour'] if cfg.run_config.use_time_encoding else []
    landmask_key = ['LANDMASK'] if cfg.run_config.use_landmask else []
    spatial_keys = ['XLAT', 'XLONG'] if cfg.run_config.use_spatial_encoding else []
    wrf_keys = ['u10', 'v10', 'T2'] + wrf_dataset.data_variables[2:] + landmask_key + spatial_keys + time_keys
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

    # metadata = test_dataset.metadata
    wrf_grid, era_grid = wrf_dataset.src_grid, era_dataset.src_grid
    era_coords = np.stack([era_grid['longitude'].flatten(), era_grid['latitude'].flatten()]).T
    wrf_coords = np.stack([wrf_grid['longitude'].flatten(), wrf_grid['latitude'].flatten()]).T
    scat_grid = sc_ds.src_grid
    scat_coords = np.stack([scat_grid['longitude'].flatten(), scat_grid['latitude'].flatten()]).T
    meaner = ClusterMapper(mapping_file=None,
                           target_coords=era_coords, input_coords=wrf_coords, 
                           weighted=cfg.run_config.weighted_meaner, 
                           save_mapping=True, save_name='meaner_mapping.npy', 
                           device=cfg.device, distance_metric='euclidean').to(cfg.device)

    stations_interpolator = InvDistTree(x=wrf_coords, q=st_ds.coords, device=cfg.device) #if False else None
    scat_interpolator = InvDistTree(x=wrf_coords, q=scat_coords, device=cfg.device) #if False else None
    # criterion = TurbulentMSE(meaner, betas, stations_interpolator, scat_interpolator, logger=logger,
    #                         kernel_type=cfg.loss_config.loss_kernel, k=cfg.loss_config.k, device=cfg.device).to(cfg.device).float()
    criterion = UpdatedTurbulentMSE(meaner, betas, stations_interpolator, scat_interpolator,
                                    logger=logger,kernel_type=cfg.loss_config.loss_kernel,
                                    k=cfg.loss_config.k, device=cfg.device).to(cfg.device).float()
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
    st_ds.wind_format = 'wd'
    results = test(model, criterion, wrf_scaler, era_scaler, test_dataloader, logger, cfg)
    print(results)


if __name__ == '__main__':
    from correction.config.cfg import cfg as default_cfg
    default_cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    main(default_cfg)