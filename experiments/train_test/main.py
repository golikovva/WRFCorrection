import sys
sys.path.insert(0, '../../')
import torch
from correction.config.config import cfg
from torch.optim import lr_scheduler
from correction.models.loss import TurbulentMSE
import os
from correction.models.changeToERA5 import MeanToERA5
from correction.data.train_test_split import split_train_val_test, find_files
from correction.data.my_dataloader import WRFDataset
from correction.data.scalers import StandardScaler
from torch.utils.data import DataLoader
from correction.test import draw_advanced_plots, test
from correction.train import train
from correction.train_and_test import train_and_test
from correction.models.build_module import build_correction_model, build_scheduler
from correction.data.logger import WRFLogger

folder_name = cfg.run_config.model_type
logger = WRFLogger(cfg.GLOBAL.MODEL_SAVE_DIR, folder_name)
# if cfg.run_config.run_mode == "test":
#     run_id = cfg.run_config.run_id
#     logger.load_configuration(run_id)

batch_size = cfg.run_config.batch_size
max_epochs = cfg.run_config.epochs
beta = 0 if cfg.run_config.run_mode == 'test' else cfg.run_config.beta
print(beta, 'beta')
use_spatiotemporal_encoding = cfg.run_config.use_spatiotemporal_encoding
use_time_encoding = cfg.run_config.use_time_encoding
LR = cfg.run_config.lr

wrf_folder = '/home/wrf_data/'
era_folder = '/home/era_data/'
stations_folder = '/home/stations/'
# wrf_folder = '/app/wrf_test_dataset'
# era_folder = '/app/era_test'
# stations_folder = 'C:\\Users\\Viktor\ml\\WRFCorrection\\stations'
# wrf_folder = 'C:\\Users\\Viktor\\Desktop\\wrf_test'
# era_folder = 'C:\\Users\\Viktor\\Desktop\\era_test'
print('Splitting train val test...')
train_files, val_files, test_files = split_train_val_test(wrf_folder, era_folder, 0.7, 0.1, 0.2)
station_files = find_files(stations_folder, '*')
print(len(station_files))
print('Split completed!')

era_scaler = StandardScaler()
wrf_scaler = StandardScaler()
era_scaler.apply_scaler_channel_params(torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_means'))[:3],
                                       torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_stds'))[:3])
wrf_scaler.apply_scaler_channel_params(torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_means')),
                                       torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_stds')))

train_dataset = WRFDataset(train_files[0], train_files[1], use_spatiotemporal_encoding=use_spatiotemporal_encoding,
                           use_time_encoding=use_time_encoding)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cfg.run_config.workers)

valid_dataset = WRFDataset(val_files[0], val_files[1], use_spatiotemporal_encoding=use_spatiotemporal_encoding,
                           use_time_encoding=use_time_encoding)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.run_config.workers)

test_dataset = WRFDataset(test_files[0], test_files[1], station_files=station_files,
                          use_spatiotemporal_encoding=use_spatiotemporal_encoding,
                          use_time_encoding=use_time_encoding)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.run_config.workers)

meaner = MeanToERA5(os.path.join(cfg.GLOBAL.BASE_DIR, 'wrferaMapping.npy'))
criterion = TurbulentMSE(meaner, beta=beta, logger=logger).to(cfg.GLOBAL.DEVICE)

model = build_correction_model(cfg.run_config.model_type)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)


if cfg.run_config.run_mode == "train+test":
    print("Started training + testing the model")
    train_and_test(train_dataloader, valid_dataloader, test_dataloader, model, optimizer, wrf_scaler, era_scaler, criterion,
                   scheduler, logger, max_epochs)
elif cfg.run_config.run_mode == "train":
    print("Started training the model")
    train(train_dataloader, valid_dataloader, model, optimizer, wrf_scaler, era_scaler, criterion,
          scheduler, logger, max_epochs)
elif cfg.run_config.run_mode == "test":
    print("Started testing the model")
    best_epoch = cfg.run_config.best_epoch_id
    state_dict = torch.load(os.path.join(logger.model_save_dir, f'model_{best_epoch}.pth'))
    model.load_state_dict(state_dict)
    save_dir = logger.save_dir
    results = test(model, criterion, wrf_scaler, era_scaler, test_dataloader, save_dir, logger)
    print(results)

