import sys

sys.path.insert(0, '../../')
# sys.path.insert(0, '/app/Precipitation-Nowcasting-master')

import torch
from torch.utils.data import DataLoader
from correction.config.config import cfg
from correction.models.loss import TurbulentMSE, TurbulentMSEMetric
import os
from experiments.net_params import encoder_params, forecaster_params

from correction.models.forecaster import Forecaster
from correction.models.encoder import Encoder
from correction.models.model import EF
from correction.models.changeToERA5 import MeanToERA5
from correction.data.train_test_split import split_train_val_test
from correction.data.scalers import StandardScaler
from correction.data.my_dataloader import WRFDataset
from correction.test import draw_advanced_plots

batch_size = 1
run_id = 4
best_epoch = 15

wrf_folder = '/home/wrf_data/'
era_folder = '/home/era_data/'
# wrf_folder = 'C:\\Users\\Viktor\\Desktop\\wrf_test'
# era_folder = 'C:\\Users\\Viktor\\Desktop\\era_test'
# wrf_folder = '/app/wrf_test_dataset'
# era_folder = '/app/era_test'
print('Splitting train val test...')
_, _, test_files = split_train_val_test(wrf_folder, era_folder, 0.7, 0.1, 0.2)
print('Split completed!')
folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]
# logger = WRFLogger(cfg.GLOBAL.MODEL_SAVE_DIR, folder_name)

era_scaler = StandardScaler()
wrf_scaler = StandardScaler()
era_scaler.apply_scaler_channel_params(torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'era_means')),
                                       torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'era_stds')))
wrf_scaler.apply_scaler_channel_params(torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_means')),
                                       torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_stds')))
use_spatiotemporal_encoding = cfg.run_config.use_spatiotemporal_encoding
test_dataset = WRFDataset(test_files[0], test_files[1],
                          use_spatiotemporal_encoding=use_spatiotemporal_encoding, file_wise=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

meaner = MeanToERA5(os.path.join(cfg.GLOBAL.BASE_DIR, 'wrferaMapping.npy'))
criterion = TurbulentMSEMetric(meaner, beta=0.5).to(cfg.GLOBAL.DEVICE)
losses = [
    TurbulentMSE(meaner, beta=2, logger=None).to(cfg.GLOBAL.DEVICE),
    TurbulentMSE(meaner, beta=0.5, logger=None).to(cfg.GLOBAL.DEVICE),
    TurbulentMSE(meaner, beta=0, logger=None).to(cfg.GLOBAL.DEVICE)
]
encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)

forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(cfg.GLOBAL.DEVICE)

model = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)
model.load_state_dict(torch.load(os.path.join(cfg.GLOBAL.BASE_DIR, f'logs/trajGRU/misc_{run_id}/models/model_{best_epoch}.pth')))
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
# loss_values = [0. for _ in range(len(losses))]

save_dir = f'logs/trajGRU/misc_{run_id}/'

if __name__ == '__main__':
    # l1, l2 = test_model(model, losses, wrf_scaler, era_scaler, test_dataloader, save_dir, draw_plots=True)
    draw_advanced_plots(model, criterion, wrf_scaler, era_scaler, test_dataset, save_dir)
    # print(l1, l2)