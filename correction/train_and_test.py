import sys

sys.path.insert(0, '../')
from correction.train import train
from correction.config.config import cfg
from correction.test import test
import torch
import os


def train_and_test(train_dataloader, valid_dataloader, test_dataloader, encoder_forecaster, optimizer, wrf_scaler,
                   era_scaler, criterion, lr_scheduler, logger, max_epochs):
    best_epoch, model = train(train_dataloader, valid_dataloader, encoder_forecaster, optimizer, wrf_scaler, era_scaler,
                              criterion, lr_scheduler, logger, max_epochs)

    state_dict = torch.load(os.path.join(logger.model_save_dir, f'model_{best_epoch}.pth'))
    model.load_state_dict(state_dict)
    save_dir = logger.save_dir
    criterion.beta = 0
    results = test(model, criterion, wrf_scaler, era_scaler, test_dataloader, save_dir, logger)
    logger.save_configuration()
    print(results)
