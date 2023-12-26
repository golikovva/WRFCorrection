import sys

sys.path.insert(0, '../')
import torch
from correction.config.config import cfg
from tqdm import tqdm
import numpy as np


def train(train_dataloader, valid_dataloader, correction_model, downscaling_model, optimizer, wrf_scaler, era_scaler,
          criterion, lr_scheduler, logger, max_epochs):

    correction_model.eval()
    downscaling_model.train()
    best_epoch = None
    try:
        for epoch in range(max_epochs):
            train_loss = train_epoch(train_dataloader, correction_model, criterion, lr_scheduler,
                                     optimizer, wrf_scaler, era_scaler, logger)
            if logger:
                logger.train_loss.append(train_loss)
            print('train loss', train_loss)
            valid_loss = eval_epoch(correction_model, criterion, wrf_scaler, era_scaler, valid_dataloader, logger,
                                    epoch)
            print('valid_loss', valid_loss)
            lr_scheduler.step()
            print(lr_scheduler.get_last_lr())
            if logger:
                best_epoch = logger.save_model(correction_model.state_dict(), epoch)
    except KeyboardInterrupt:
        pass
    logger.save_configuration()
    return best_epoch, correction_model, downscaling_model


def train_epoch(dataloader, correction_model, downscaling_model, criterion, lr_scheduler, optimizer, wrf_scaler,
                era_scaler, logger=None):
    train_loss = 0
    downscaling_model.train()

    for train_data, _, station, i in (pbar := tqdm(dataloader)):
        train_data = torch.swapaxes(train_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
        station = torch.swapaxes(station[:, :, [1, 3]].type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
        train_data = wrf_scaler.channel_transform(train_data, 2)

        optimizer.zero_grad()

        output = correction_model(train_data)
        output = wrf_scaler.channel_inverse_transform(output, 2)

        station_pred = downscaling_model(output)

        loss = criterion(station_pred, station)
        # loss = criterion(train_data, output, train_label)  # , mask)
        loss.backward()

        torch.nn.utils.clip_grad_value_(downscaling_model.parameters(), clip_value=50.0)
        optimizer.step()

        l = loss.detach().item()
        train_loss += l
        pbar.set_description(f'{l}')

    return train_loss / len(dataloader)


def eval_epoch(correction_model, downscaling_model, criterion, wrf_scaler, era_scaler, dataloader, logger, epoch=None):
    with torch.no_grad():
        correction_model.eval()
        downscaling_model.eval()
        valid_loss = 0.0
        for valid_data, _, station in tqdm(dataloader):
            valid_data = torch.swapaxes(valid_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            station = torch.swapaxes(station[:, :, [1, 3]].type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            valid_data = wrf_scaler.channel_transform(valid_data, 2)

            output = correction_model(valid_data)
            output = wrf_scaler.channel_inverse_transform(output, 2)

            station_pred = downscaling_model(output)
            loss = criterion(station_pred, station, logger)
            valid_loss += loss.item()
        valid_loss = valid_loss / len(dataloader)
        if logger:
            logger.print_stat_readable(epoch)
    return valid_loss
