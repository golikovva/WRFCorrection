import os
import sys
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import optuna

import logging
from optuna.trial import TrialState

sys.path.insert(0, '../../')

from correction.config.cfg import cfg
from correction.models.loss import TurbulentMSE
from correction.models.changeToERA5 import MeanToERA5
from correction.data.train_test_split import split_train_val_test, find_files
from correction.data.my_dataloader import WRFDataset, custom_collate, TestSampler
from correction.data.scaler import StandardScaler
from correction.data.logger import WRFLogger
from correction.pipeline.train import trial_model
from correction.models.build_module import build_correction_model
from correction.helpers.interpolation import InvDistTree

cfg['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder_name = cfg.model_type
study_name = 'mesoscale_parameters_study'  # Unique identifier of the study.


def objective(trial):
    logger = WRFLogger(cfg.data.logs_folder, os.path.join(folder_name, 'hyperparameters_optimization', study_name))

    beta1 = trial.suggest_categorical("beta1", [0, 1])
    beta2 = trial.suggest_categorical("beta2", [0, 0.1, 1, 5, 10, 40])
    mesoscale_k = trial.suggest_categorical('k', [3, 5, 7, 9])
    beta3_t2 = trial.suggest_float("beta3_t2", 0, 0.2)  # границы 15 лучших запусков
    beta3_w10 = trial.suggest_float("beta3_w10", 0, 0.2)  # границы 15 лучших запусков
    beta4 = trial.suggest_float("beta4", 0, 2)
    betas = [beta1, beta2, [beta3_w10, beta3_t2], beta4]

    print('Splitting train val test...')
    train_files, val_files, test_files = split_train_val_test(cfg.data.wrf_folder, cfg.data.era_folder, 0.7, 0.1, 0.2)
    station_files = find_files(cfg.data.stations_folder, '*.pkl')
    scatter_files = find_files(cfg.data.scatter_folder, '*')

    print('Split completed!')

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    train_dataset = WRFDataset(train_files[0], train_files[1], wrf_variables=cfg.data.wrf_variables,
                               station_files=station_files, scatter_files=scatter_files,
                               use_spatial_encoding=cfg.run_config.use_spatial_encoding,
                               use_time_encoding=cfg.run_config.use_time_encoding,
                               use_landmask=cfg.run_config.use_landmask)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.run_config.batch_size, shuffle=True,
                                  num_workers=cfg.run_config.num_workers,
                                  collate_fn=custom_collate, pin_memory=True)
    print(len(train_dataloader), 'train dataloader length')
    test_dataset = WRFDataset(test_files[0], test_files[1], wrf_variables=cfg.data.wrf_variables,
                              station_files=station_files, scatter_files=scatter_files,
                              use_spatial_encoding=cfg.run_config.use_spatial_encoding,
                              use_time_encoding=cfg.run_config.use_time_encoding,
                              use_landmask=cfg.run_config.use_landmask)
    test_sampler = TestSampler(len(test_dataset), 4)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg.run_config.num_workers,
                                 collate_fn=custom_collate, sampler=test_sampler)
    metadata = train_dataset.metadata

    scaler_type = trial.suggest_categorical("scaler_type", ['seasonal_mean_std', 'std_only', 'ordinary'])
    loss_type = trial.suggest_categorical("consider_channel_dispersion", ['weighted_loss', 'ordinary'])
    means_dict = torch.load(cfg.data.wrf_mean_path)
    stds_dict = torch.load(cfg.data.wrf_std_path)
    time_keys = ['day', 'hour'] if cfg.run_config.use_spatial_encoding else []
    landmask_key = ['LANDMASK'] if cfg.run_config.use_landmask else []
    wrf_keys = ['u10', 'v10', 'T2'] + test_dataset.wrf_variables[2:] + landmask_key + time_keys
    era_keys = ['u10', 'v10', 'T2']
    print(wrf_keys, '- wrf channels to transform')
    print(era_keys, '- era channels to transform')
    era_scaler = StandardScaler()
    wrf_scaler = StandardScaler()
    era_scaler.apply_scaler_channel_params(torch.tensor([means_dict[x] for x in era_keys]).float().to(cfg.device),
                                           torch.tensor([stds_dict[x] for x in era_keys]).float().to(cfg.device))
    wrf_scaler.apply_scaler_channel_params(torch.tensor([means_dict[x] for x in wrf_keys]).float().to(cfg.device),
                                           torch.tensor([stds_dict[x] for x in wrf_keys]).float().to(cfg.device))

    era_coords = np.stack([metadata['era_xx'].flatten(), metadata['era_yy'].flatten()]).T
    wrf_coords = np.stack([metadata['wrf_xx'].flatten(), metadata['wrf_yy'].flatten()]).T
    scat_coords = np.stack([metadata['scat_xx'].flatten(), metadata['scat_yy'].flatten()]).T if scatter_files else None
    meaner = MeanToERA5(os.path.join(cfg.data.logs_folder, 'wrferaMapping.npy'),
                        era_coords=era_coords, wrf_coords=wrf_coords,
                        weighted=cfg.run_config.weighted_meaner) \
        .to(cfg.device)
    stations_interpolator = InvDistTree(x=wrf_coords, q=metadata['coords']) if station_files else None
    scat_interpolator = InvDistTree(x=wrf_coords, q=scat_coords) if scatter_files else None

    print(betas, 'betas')

    criterion = TurbulentMSE(meaner, betas, stations_interpolator, scat_interpolator,
                             kernel_type=cfg.run_config.loss_kernel, k=mesoscale_k).to(cfg.device)

    # model_type = trial.suggest_categorical("model_type", ['BERTunet', 'Identity'])
    model_type = cfg.run_config.model_type
    model = build_correction_model(model_type)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.2)
    print("Started optimizing model parameters")
    acc = trial_model(train_dataloader, test_dataloader, model, optimizer, wrf_scaler, era_scaler, criterion,
                      scheduler, logger=logger, max_epochs=3, trial=trial)
    print(acc, 'accuracy')
    return acc


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
storage_name = "sqlite:///{}.db".format(os.path.join(cfg.data.logs_folder, study_name))
directions = ['minimize' for i in range(3 + 2 + 2 + 1)]  # minimize for each channel of each train data type
study = optuna.create_study(directions=directions, study_name=study_name,
                            storage=storage_name, load_if_exists=True)
n_trials = 24
for i in range(n_trials):
    study.enqueue_trial({'beta1': 1., 'beta3_t2': 0.03, 'beta3_w10': 0.06, 'beta4': 1.3,
                         "scaler_type": 'ordinary', "consider_channel_dispersion": 'ordinary'})
study.optimize(objective, n_trials=n_trials)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

df = study.trials_dataframe()
print(df)
df.to_csv(os.path.join(cfg.data.logs_folder, study_name))
