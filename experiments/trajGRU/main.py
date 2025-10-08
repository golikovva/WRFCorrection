import sys
sys.path.insert(0, '../../')
import torch
from correction.config.config import cfg
from correction.models.forecaster import Forecaster
from correction.models.encoder import Encoder
from correction.models.model import EF
from torch.optim import lr_scheduler
from correction.models.loss import TurbulentMSE
import os
from experiments.net_params import encoder_params, forecaster_params
from correction.models.changeToERA5 import MeanToERA5
from correction.data.train_test_split import split_train_val_test
from correction.data.my_dataloader import WRFDataset
from correction.data.scalers import StandardScaler
from torch.utils.data import DataLoader
from correction.train import train
from correction.data.logger import WRFLogger

batch_size = 16
max_epochs = 50
use_spatiotemporal_encoding = cfg.run_config.use_spatiotemporal_encoding
LR = 1e-4
wrf_folder = '/home/wrf_data/'
era_folder = '/home/era_data/'
# wrf_folder = '/app/wrf_test_dataset'
# era_folder = '/app/era_test'
print('Splitting train val test...')
train_files, val_files, test_files = split_train_val_test(wrf_folder, era_folder, 0.7, 0.1, 0.2)
print('Split completed!')
folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]
logger = WRFLogger(cfg.GLOBAL.MODEL_SAVE_DIR, folder_name)

era_scaler = StandardScaler()
wrf_scaler = StandardScaler()
era_scaler.apply_scaler_channel_params(torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'era_means')),
                                       torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'era_stds')))
wrf_scaler.apply_scaler_channel_params(torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_means')),
                                       torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_stds')))

train_dataset = WRFDataset(train_files[0], train_files[1], use_spatiotemporal_encoding=use_spatiotemporal_encoding)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

valid_dataset = WRFDataset(val_files[0], val_files[1], use_spatiotemporal_encoding=use_spatiotemporal_encoding)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

meaner = MeanToERA5(os.path.join(cfg.GLOBAL.BASE_DIR, 'wrferaMapping.npy'))
criterion = TurbulentMSE(meaner, beta=0.5, logger=logger).to(cfg.GLOBAL.DEVICE)

encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)

forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(cfg.GLOBAL.DEVICE)

encoder_forecaster = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)

optimizer = torch.optim.Adam(encoder_forecaster.parameters(), lr=LR, weight_decay=1e-5)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=0.1)
mult_step_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

train(train_dataloader, valid_dataloader, encoder_forecaster, optimizer, wrf_scaler, era_scaler, criterion,
      mult_step_scheduler, logger, folder_name, max_epochs)
