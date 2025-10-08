import sys
sys.path.insert(0, '../../')
import torch
import numpy as np
from correction.config.cfg import cfg
from torch.optim import lr_scheduler
from correction.models.loss import TurbulentMSE
import os
from correction.models.changeToERA5 import MeanToERA5
from correction.data.train_test_split import split_train_val_test, find_files, split_dates_by_dates, split_dates
from correction.data.my_dataloader import WRFDataset, none_consistent_collate, TestSampler
from correction.data.data_utils import IFSs2sDataset, ERAMonthlyDataset, StackDataset, \
    dataset_with_indices, Sampler, StationsDataset, ScatterDataset, StackDataset, variable_len_collate, StationsNoneDataset, ScatterNoneDataset
from correction.data.scaler import StandardScaler
from torch.utils.data import DataLoader
from correction.pipeline.test import test
from correction.pipeline.train import train
from correction.models.build_module import build_correction_model
from correction.data.logger import WRFLogger
from correction.helpers.interpolation import InvDistTree



def main(cfg):
    folder_name = 'ifs_finetune'
    logger = WRFLogger(cfg.data.logs_folder, folder_name)

    betas = [cfg.betas.beta1, cfg.betas.beta2, [cfg.betas.beta3_w10, cfg.betas.beta3_t2], cfg.betas.beta4]
    print(betas, 'betas')

    print('Splitting train val test...')
    max_sl = cfg.s2s.sequence_len
    ifs_dataset = IFSs2sDataset('/home/ifs_data/', data_variables=['VAR_10U', 'VAR_10V', 'VAR_2T', 'CI', 'latitude', 'longitude', 'LSM'], seq_len=max_sl, add_time_encoding=True, add_coords=False)
    era_dataset = ERAMonthlyDataset('/home/era_data_wp/', cfg.data.era_variables, seq_len=max_sl, time_resolution_h=6)
    st_ds = StationsNoneDataset()
    sc_ds = ScatterNoneDataset()
    dataset = dataset_with_indices(StackDataset)(ifs_dataset, era_dataset, st_ds, sc_ds,)

    start_date = np.datetime64('2021-01-01T06')
    train_days = np.arange(start_date, start_date+np.timedelta64(365,'D')-np.timedelta64(max_sl*6, 'h'), np.timedelta64(6, 'h'), dtype='datetime64[h]')
    val_days = np.arange(start_date+np.timedelta64(365,'D'), start_date+np.timedelta64(365*2,'D')-np.timedelta64(max_sl*6, 'h'), np.timedelta64(6, 'h'), dtype='datetime64[h]')
    test_days = np.arange(start_date+np.timedelta64(365*2,'D'), start_date+np.timedelta64(365*3,'D')-np.timedelta64(max_sl*6, 'h'), np.timedelta64(6*max_sl, 'h'), dtype='datetime64[h]')
    print('Split completed!')
    print(len(train_days), len(val_days), len(test_days))
    print(f'from {train_days[0].astype(str)} to {train_days[-1].astype(str)} is a train period')
    print(f'from {val_days[0].astype(str)} to {val_days[-1].astype(str)} is a val period')
    print(f'from {test_days[0].astype(str)} to {test_days[-1].astype(str)} is a test period')
    train_sampler = Sampler(train_days, shuffle=True)
    val_sampler = Sampler(val_days, shuffle=False)
    test_sampler = Sampler(test_days, shuffle=False)
    collate_fn = variable_len_collate if cfg.run_config.variable_sequence_length else none_consistent_collate
    train_dataloader = DataLoader(dataset, batch_size=cfg.run_config.batch_size, num_workers=cfg.run_config.num_workers,
                                  sampler=train_sampler, collate_fn=collate_fn, pin_memory=True)
    valid_dataloader = DataLoader(dataset, batch_size=cfg.run_config.batch_size, num_workers=cfg.run_config.num_workers,
                                  sampler=val_sampler, collate_fn=collate_fn, pin_memory=True)
    test_dataloader = DataLoader(dataset, batch_size=1, num_workers=cfg.run_config.num_workers,
                                 sampler=test_sampler, collate_fn=collate_fn, pin_memory=True)

    means_dict = torch.load(cfg.data.wrf_mean_path+'_1')
    stds_dict = torch.load(cfg.data.wrf_std_path+'_1')
    time_keys = ['day', 'hour'] if cfg.run_config.use_spatial_encoding else []
    landmask_key = ['LANDMASK'] if cfg.run_config.use_landmask else []
    wrf_keys = ['u10', 'v10', 'T2'] + cfg.data.wrf_variables[2:] + ['XLAT_WP', 'XLONG_WP']  + landmask_key + time_keys
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
    wrf_grid, era_grid = ifs_dataset.src_grid, era_dataset.src_grid
    era_coords = np.stack([era_grid['longitude'].flatten(), era_grid['latitude'].flatten()]).T
    wrf_coords = np.stack([wrf_grid['longitude'].flatten(), wrf_grid['latitude'].flatten()]).T

    scat_coords = np.stack([metadata['scat_xx'].flatten(), metadata['scat_yy'].flatten()]).T if False else None
    meaner = MeanToERA5(mapping_file=None,
                        era_coords=era_coords, wrf_coords=wrf_coords,                           # todo fix backward dtype error !!!
                        weighted=False).to(cfg.device)
    stations_interpolator = InvDistTree(x=wrf_coords, q=metadata['coords'], device=cfg.device) if False else None
    scat_interpolator = InvDistTree(x=wrf_coords, q=scat_coords, device=cfg.device) if False else None
    criterion = TurbulentMSE(meaner, betas, stations_interpolator, scat_interpolator, logger=logger,
                            kernel_type=cfg.loss_config.loss_kernel, k=cfg.loss_config.k, device=cfg.device).to(cfg.device).float()

    model = build_correction_model(cfg)
    model_dict=model.state_dict()

    pretrained_dict = torch.load(os.path.join('/home/logs/BERTunet/misc_110/models', f'model_{5}.pth'))
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'patch_encoding' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.2)


    best_epoch = cfg.test_config.best_epoch_id
    if 'train' in cfg.run_config.run_mode:
        print(f"Started training the model: run no {logger.experiment_number}")
        best_epoch, encoder_forecaster = train(train_dataloader, valid_dataloader, model, optimizer, wrf_scaler,
                                            era_scaler, criterion, scheduler, logger, cfg)

    print(f"Started testing the model: run no {logger.experiment_number}")
    state_dict = torch.load(os.path.join(logger.model_save_dir, f'model_{best_epoch}.pth'))
    model.load_state_dict(state_dict)
    results = test(model, criterion, wrf_scaler, era_scaler, test_dataloader, logger, cfg)
    print(results)


if __name__ == '__main__':
    cfg['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(cfg)