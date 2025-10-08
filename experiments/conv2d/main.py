import sys

sys.path.insert(0, '../../')
# sys.path.insert(0, '/app/Precipitation-Nowcasting-master')

import torch
from torch.utils.data import DataLoader
from correction.config.config import cfg
from torch.optim import lr_scheduler
from correction.models.loss import TurbulentMSE

from correction.train import train
import os

from experiments.net_params import conv2d_params
from correction.models.model import Predictor
from correction.models.changeToERA5 import MeanToERA5
from correction.data.train_test_split import split_train_val_test
from correction.data.logger import WRFLogger
from correction.data.scalers import StandardScaler
from correction.data.my_dataloader import WRFDataset

batch_size = 32
max_epochs = 100

LR = 1e-3

wrf_folder = '/home/wrf_data/'
era_folder = '/home/era_data/'
# wrf_folder = 'C:\\Users\\Viktor\\Desktop\\wrf_test'
# era_folder = 'C:\\Users\\Viktor\\Desktop\\era_test'
# wrf_folder = '/app/wrf_test_dataset'
# era_folder = '/app/era_test'
print('Splitting train val test...')
train_files, val_files, test_files = split_train_val_test(wrf_folder, era_folder, 0.7, 0.1, 0.2)
print('Split completed!')
folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]
logger = WRFLogger(cfg.GLOBAL.MODEL_SAVE_DIR, folder_name)

era_scaler = StandardScaler()
wrf_scaler = StandardScaler()
era_scaler.apply_scaler_channel_params([torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'era_means'))[2]],
                                       [torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'era_stds'))[2]])
wrf_scaler.apply_scaler_channel_params([torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_means'))[2]],
                                       [torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_stds'))[2]])

train_dataset = WRFDataset(train_files[0], train_files[1], wrf_variables=['T2'], era_variables=['t2m'])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

valid_dataset = WRFDataset(val_files[0], val_files[1], wrf_variables=['T2'], era_variables=['t2m'])
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=12)

meaner = MeanToERA5(os.path.join(cfg.GLOBAL.BASE_DIR, 'wrferaMapping.npy'))
criterion = TurbulentMSE(meaner, beta=0.5, logger=None).to(cfg.GLOBAL.DEVICE)

model = Predictor(conv2d_params).to(cfg.GLOBAL.DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


if __name__ == '__main__':
    train(train_dataloader, valid_dataloader, model, optimizer, wrf_scaler, era_scaler,
          criterion, exp_lr_scheduler, logger, folder_name, max_epochs)
