import sys

sys.path.insert(0, '../')
import torch
from correction.config.config import cfg
from tqdm import tqdm


# era_folder = 'C:\\Users\\Viktor\\Desktop\\era_test'
# wrf_folder = 'C:\\Users\\Viktor\\Desktop\\wrf_test'
# wrf_folder = '/app/wrf_test_dataset/'
# era_folder = '/app/era_test/'


def train(train_dataloader, valid_dataloader, encoder_forecaster, optimizer, wrf_scaler, era_scaler,
          criterion, lr_scheduler, logger, max_epochs):
    best_epoch = None
    try:
        for epoch in range(max_epochs):
            train_loss = train_epoch(train_dataloader, encoder_forecaster, criterion, lr_scheduler,
                                     optimizer, wrf_scaler, era_scaler, logger)
            if logger:
                logger.train_loss.append(train_loss)
            print('train loss', train_loss)
            valid_loss = eval_epoch(encoder_forecaster, criterion, wrf_scaler, era_scaler, valid_dataloader, logger, epoch)
            print('valid_loss', valid_loss)
            lr_scheduler.step()
            print(lr_scheduler.get_last_lr())
            if logger:
                best_epoch = logger.save_model(encoder_forecaster.state_dict(), epoch)
    except KeyboardInterrupt:
        pass
    logger.save_configuration()
    return best_epoch, encoder_forecaster


def train_epoch(dataloader, model, criterion, lr_scheduler, optimizer, wrf_scaler, era_scaler, logger=None):
    train_loss = 0
    model.train()

    for train_data, train_label, stations, _ in (pbar := tqdm(dataloader)):
        train_data = torch.swapaxes(train_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
        train_label = torch.swapaxes(train_label.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
        stations = torch.permute(stations.type(torch.float).to(cfg.GLOBAL.DEVICE), (1, 0, 3, 2))
        train_data = wrf_scaler.channel_transform(train_data, 2)
        train_label = era_scaler.channel_transform(train_label, 2)
        # stations = scaler.channel_transform(stations, 2)

        optimizer.zero_grad()

        output = model(train_data)
        if cfg.run_config.use_spatiotemporal_encoding:
            train_data = train_data[:, :, :3]
        loss = criterion(train_data, output, train_label, stations[..., [3, 1], :], wrf_scaler)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=50.0)
        optimizer.step()

        l = loss.detach().item()
        train_loss += l
        pbar.set_description(f'{l}')

    return train_loss / len(dataloader)


def eval_epoch(model, criterion, wrf_scaler, era_scaler, dataloader, logger, epoch=None):
    with torch.no_grad():
        model.eval()
        valid_loss = 0.0
        for valid_data, valid_label, stations, _ in tqdm(dataloader):
            valid_data = torch.swapaxes(valid_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            valid_label = torch.swapaxes(valid_label.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            stations = torch.permute(stations.type(torch.float).to(cfg.GLOBAL.DEVICE), (1, 0, 3, 2))
            valid_data = wrf_scaler.channel_transform(valid_data, 2)
            valid_label = era_scaler.channel_transform(valid_label, 2)

            output = model(valid_data)
            if cfg.run_config.use_spatiotemporal_encoding:
                valid_data = valid_data[:, :, :3]
                print(valid_data.shape, output.shape, stations[..., [3, 1], :].shape)
            loss = criterion(valid_data, output, valid_label, stations[..., [3, 1], :], wrf_scaler, logger)
            valid_loss += loss.item()
        valid_loss = valid_loss / len(dataloader)
        if logger:
            logger.print_stat_readable(epoch)
    return valid_loss
