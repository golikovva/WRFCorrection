import sys

sys.path.insert(0, '../../')
# sys.path.insert(0, '/app/Precipitation-Nowcasting-master')

import torch
from torch.utils.data import DataLoader
from correction.config.config import cfg
from correction.models.loss import TurbulentMSE
import os

from experiments.net_params import conv2d_params
from correction.models.model import Predictor
from correction.models.changeToERA5 import MeanToERA5
from correction.data.train_test_split import split_train_val_test
from correction.data.scalers import StandardScaler
from correction.data.my_dataloader import WRFDataset
from correction.test import test_model

batch_size = 32
run_id = 25
best_epoch = 32
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
era_scaler.apply_scaler_channel_params([torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'era_means'))[2]],
                                       [torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'era_stds'))[2]])
wrf_scaler.apply_scaler_channel_params([torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_means'))[2]],
                                       [torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_stds'))[2]])

test_dataset = WRFDataset(test_files[0][:32], test_files[1][:32], wrf_variables=['T2'], era_variables=['t2m'])
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

meaner = MeanToERA5(os.path.join(cfg.GLOBAL.BASE_DIR, 'wrferaMapping.npy'))
criterion = TurbulentMSE(meaner, beta=0.5, logger=None).to(cfg.GLOBAL.DEVICE)
model = Predictor(conv2d_params).to(cfg.GLOBAL.DEVICE)
model.load_state_dict(torch.load(os.path.join(cfg.GLOBAL.BASE_DIR, f'logs/conv2d/misc_{run_id}/models/model_{best_epoch}.pth')))
# loss_values = [0. for _ in range(len(losses))]


if __name__ == '__main__':
    l1, l2 = test_model(model, criterion, wrf_scaler, era_scaler, test_dataloader, draw_plots=True)
    print(l1, l2)
