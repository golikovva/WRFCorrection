import sys

sys.path.insert(0, '../../')
import torch
import numpy as np
from correction.config.cfg import cfg
from torch.optim import lr_scheduler
from correction.models.loss import WindLoss
import os
from correction.data.train_test_split import split_train_val_test, find_files, split_dates_by_dates, split_dates
from correction.data.my_dataloader import WRFDataset, none_consistent_collate, TestSampler
from correction.data.data_utils import WRFs2sDataset, ERAs2sDataset, StationsDataset, ScatterDataset, StackDataset, \
    dataset_with_indices, Sampler, variable_len_collate, ScatterNoneDataset, StationsNoneDataset
from correction.data.scaler import StandardScaler
from torch.utils.data import DataLoader
from experiments.trajgru_baseline.test import uv_test
from experiments.trajgru_baseline.new_test import test
from correction.pipeline.train import train
from correction.models.build_module import build_correction_model
from correction.data.logger import WRFLogger
from correction.helpers.interpolation import InvDistTree
from correction.models.changeToERA5 import MeanToERA5, ClusterMapper


def main(cfg):
    cfg['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cfg['device'] = torch.device("cpu")

    cfg['model_type'] = 'DETrajGRU'
    interpolation_type = 'upsampling'
    wrf_variables = ['uvmet10']
    era_variables = ['u10', 'v10']

    folder_name = cfg.model_type
    logger = WRFLogger(cfg, cfg.data.logs_folder, folder_name)

    wrf_dataset = WRFs2sDataset(cfg.data.wrf_folder, wrf_variables, seq_len=8)
    era_dataset = ERAs2sDataset(cfg.data.era_folder, era_variables, seq_len=8)
    st_ds, sc_ds = StationsNoneDataset(), ScatterNoneDataset()
    # st_ds = StationsDataset(cfg.data.stations_folder, data_variables=[], seq_len=8)
    # sc_ds = ScatterDataset(cfg.data.scatter_folder, seq_len=8)
    dataset = dataset_with_indices(StackDataset)(wrf_dataset, era_dataset, st_ds, sc_ds)

    start_date, end_date = np.datetime64(cfg.data.start_date), np.datetime64(cfg.data.end_date)
    train_days, val_days, test_days = split_dates(start_date, end_date, 0.7, 0.1, 0.2)
    train_days = train_days[
        ~((train_days >= np.datetime64('2020-12-31T00')) & (train_days <= np.datetime64('2020-12-31T23')))]

    train_sampler = Sampler(train_days, shuffle=True)
    val_sampler = Sampler(val_days, shuffle=False)
    test_sampler = Sampler(test_days, shuffle=False)

    train_dataloader = DataLoader(dataset, batch_size=cfg.run_config.batch_size, num_workers=cfg.run_config.num_workers,
                                sampler=train_sampler, collate_fn=variable_len_collate, pin_memory=True)
    valid_dataloader = DataLoader(dataset, batch_size=cfg.run_config.batch_size, num_workers=cfg.run_config.num_workers,
                                sampler=val_sampler, collate_fn=variable_len_collate, pin_memory=True)


    means_dict = torch.load(cfg.data.wrf_mean_path)
    stds_dict = torch.load(cfg.data.wrf_std_path)
    wrf_keys = ['u10', 'v10']
    era_keys = ['u10', 'v10']

    era_scaler = StandardScaler()
    wrf_scaler = StandardScaler()
    era_scaler.apply_scaler_channel_params(torch.tensor([means_dict[x] for x in era_keys]).float().to(cfg.device),
                                        torch.tensor([stds_dict[x] for x in era_keys]).float().to(cfg.device))
    wrf_scaler.apply_scaler_channel_params(torch.tensor([means_dict[x] for x in wrf_keys]).float().to(cfg.device),
                                        torch.tensor([stds_dict[x] for x in wrf_keys]).float().to(cfg.device))
    print(wrf_scaler.means, wrf_scaler.stddevs)

    wrf_grid, era_grid = wrf_dataset.src_grid, era_dataset.src_grid
    era_coords = np.stack([era_grid['longitude'].flatten(), era_grid['latitude'].flatten()]).T
    wrf_coords = np.stack([wrf_grid['longitude'].flatten(), wrf_grid['latitude'].flatten()]).T
    meaner = ClusterMapper(mapping_file=None,
                           target_coords=era_coords, input_coords=wrf_coords, 
                           weighted=cfg.run_config.weighted_meaner, 
                           save_mapping=True, save_name='meaner_mapping.npy', 
                           device=cfg.device, distance_metric='euclidean').to(cfg.device)
    interpolator = InvDistTree(x=era_coords, q=wrf_coords, n_near=4, device=cfg.device)
    loss_func = WindLoss(interpolator, device=cfg.device)

    model = build_correction_model(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.2)

    best_epoch = cfg.test_config.best_epoch_id
    if 'train' in cfg.run_config.run_mode:
        print(f"Started training the model: run no {logger.experiment_number}")
        best_epoch, encoder_forecaster = train(train_dataloader, valid_dataloader, model, optimizer, wrf_scaler,
                                            era_scaler, loss_func, scheduler, logger, cfg)

    print(f"Started testing the model: run no {logger.experiment_number}")
    state_dict = torch.load(os.path.join(logger.model_save_dir, f'model_{best_epoch}.pth'))
    model.load_state_dict(state_dict)

    st_ds = StationsDataset(cfg.data.stations_folder, data_variables=[], seq_len=8)
    sc_ds = ScatterDataset(cfg.data.scatter_folder, seq_len=8)
    dataset = dataset_with_indices(StackDataset)(wrf_dataset, era_dataset, st_ds, sc_ds)
    st_ds.wind_format = 'wd'
    test_dataloader = DataLoader(dataset, batch_size=cfg.run_config.batch_size, num_workers=cfg.run_config.num_workers,
                                 sampler=test_sampler, collate_fn=variable_len_collate, pin_memory=True)
    results = test(model, meaner, wrf_scaler, era_scaler, test_dataloader, logger, cfg)
    # _, _, test_files = split_train_val_test(cfg.data.wrf_folder, cfg.data.era_folder, 0.7, 0.1, 0.2)
    # station_files = find_files(cfg.data.stations_folder, '*.pkl')
    # scatter_files = find_files(cfg.data.scatter_folder, '*')
    # test_dataset = WRFDataset(test_files[0], test_files[1], wrf_variables=wrf_variables, era_variables=era_variables,
    #                         station_files=station_files, scatter_files=scatter_files, seq_len=8,
    #                         use_spatial_encoding=False,
    #                         use_time_encoding=False,
    #                         use_landmask=False)
    # test_sampler = TestSampler(len(test_dataset), 4)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg.run_config.num_workers,
    #                             collate_fn=none_consistent_collate, sampler=test_sampler)

    # from correction.models.loss import TurbulentMSE
    # from correction.models.changeToERA5 import MeanToERA5
    # betas = [cfg.betas.beta1, cfg.betas.beta2, [cfg.betas.beta3_w10, cfg.betas.beta3_t2], cfg.betas.beta4]
    # metadata = test_dataset.metadata
    # era_coords = np.stack([metadata['era_xx'].flatten(), metadata['era_yy'].flatten()]).T
    # wrf_coords = np.stack([metadata['wrf_xx'].flatten(), metadata['wrf_yy'].flatten()]).T
    # scat_coords = np.stack([metadata['scat_xx'].flatten(), metadata['scat_yy'].flatten()]).T if scatter_files else None
    # meaner = MeanToERA5(os.path.join(cfg.data.logs_folder, 'wrferaMapping.npy'),
    #                     era_coords=era_coords, wrf_coords=wrf_coords,
    #                     weighted=cfg.run_config.weighted_meaner) \
    #     .to(cfg.device)
    # stations_interpolator = InvDistTree(x=wrf_coords, q=metadata['coords'],
    #                                     device=cfg.device) if station_files else None
    # scat_interpolator = InvDistTree(x=wrf_coords, q=scat_coords, device=cfg.device) if scatter_files else None
    # criterion = TurbulentMSE(meaner, betas, stations_interpolator, scat_interpolator, logger=logger,
    #                         kernel_type=cfg.loss_config.loss_kernel, channels=2,
    #                         k=cfg.loss_config.k, device=cfg.device).to(
    #     cfg.device)


    # results = uv_test(model, criterion, wrf_scaler, era_scaler, test_dataloader, logger, cfg)
    print(results)

if __name__ == '__main__':
    cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    main(cfg)