
# Neural Network Atmospheric Bias Correction

Official implementation for the paper:

**Neural network atmospheric bias correction on heterogeneous data with fine-scale dynamics preservation**  
Viktor Golikov, Mikhail Krinitskiy, Alexander Gavrikov, Evgeny Burnaev, Vladimir Vanovskiy  
*Quarterly Journal of the Royal Meteorological Society*, 2026, e70209  
DOI: [10.1002/qj.70209](https://doi.org/10.1002/qj.70209)

## Overview

This repository contains the code for neural-network-based statistical bias correction of high-resolution WRF forecasts over the Kara and Barents Seas.

The method is designed to reduce systematic errors in near-surface atmospheric variables while preserving fine-scale atmospheric structures present in the original high-resolution WRF fields. The correction model is based on a U-net architecture with Transformer attention in the latent space, referred to in the paper as **BERTUnet**.

The model is trained using heterogeneous reference data sources, including:

- ERA5 reanalysis;
- ground-based meteorological station observations;
- satellite scatterometer wind measurements.

The main idea of the method is to correct large-scale forecast biases while explicitly preserving mesoscale dynamics through a dedicated fine-scale-preserving loss term.

## Repository structure

```text
WRFCorrection/
├── correction/
│   ├── config/
│   │   ├── config.yaml        # Main experiment configuration
│   │   └── cfg.py             # Config loader
│   ├── data/                  # Dataset and dataloader utilities
│   ├── helpers/               # Interpolation and helper routines
│   ├── models/                # Correction models and loss functions
│   └── pipeline/              # Training and testing pipelines
├── experiments/
│   └── train_test/
│       └── main.py            # Main training/testing entry point
├── Dockerfile
└── README.md
````

## Installation

The recommended way to run the code is to use Docker.

Build the image:

```bash
git clone https://github.com/golikovva/WRFCorrection.git
cd WRFCorrection

docker build -t wrf-correction .
```

Run the container:

```bash
docker run -it --rm \
    --name wrf_correction \
    --gpus all \
    --ipc=host \
    -v /path/to/wrf/files:/home/wrf_data \
    -v /path/to/era5/files:/home/era_data \
    -v /path/to/scatterometer/files:/home/scatter \
    -v /path/to/station/files:/home/stations/interp \
    -v /path/to/metadata:/home/metadata \
    -v $(pwd)/logs:/home/logs \
    wrf-correction
```

Inside the container, the working directory is:

```bash
/home/experiments/train_test
```

## Data

The training pipeline expects preprocessed WRF, ERA5, station, scatterometer, and metadata files.

By default, the paths are configured in:

```text
correction/config/config.yaml
```

The default paths used inside the Docker container are:

```yaml
data:
  wrf_folder: '/home/wrf_data/'
  era_folder: '/home/era_data/'
  scatter_folder: '/home/scatter/'
  stations_folder: '/home/stations/interp/'
  wrf_mean_path: "/home/metadata/means_dict"
  wrf_std_path: "/home/metadata/stds_dict"
  logs_folder: "/home/logs/"
```

The default study period in the published experiments is:

```yaml
data:
  start_date: '2019-01-01'
  end_date: '2023-08-09'
```

The main corrected WRF variables are:

```yaml
wrf_variables: ['uvmet10', 'T2', 'SEAICE', 'HGT']
```

The target ERA5 variables are:

```yaml
era_variables: ['u10', 'v10', 't2m']
```

The data used in the paper are available from the corresponding author upon reasonable request, as stated in the paper.

## Configuration

The main configuration file is:

```text
correction/config/config.yaml
```

Important fields include:

```yaml
model_type: BERTunet_raw

train:
  max_epochs: 15
  scheduler_type: MultiStepLR
  lr: 0.0001

run_config:
  run_mode: 'train'
  num_workers: 16
  batch_size: 16
  use_spatial_encoding: 1
  use_time_encoding: 1
  use_landmask: 0
  weighted_meaner: 1

loss_config:
  loss_kernel: 'gauss'
  k: 9
```

The loss weights are controlled by:

```yaml
betas:
  beta1: 1
  beta2: 0
  beta3_t2: 0
  beta3_w10: 0
  beta4: 0
```

These coefficients control the relative contribution of different data sources and the mesoscale-preserving regularization term.

## Training

To train the model, first edit:

```text
correction/config/config.yaml
```

Set:

```yaml
run_config:
  run_mode: 'train'
```

Then run:

```bash
cd experiments/train_test
python main.py
```

During training, the code will:

1. split the available WRF and ERA5 files into train, validation, and test subsets;
2. load station and scatterometer data if available;
3. initialize the WRF and ERA5 scalers from the metadata files;
4. construct the WRF-to-ERA5 interpolation operator;
5. build the correction model;
6. train the model;
7. save checkpoints and the used configuration to the log directory;
8. run testing using the best checkpoint.

Training outputs are written to:

```text
/home/logs/
```

or to the path specified by:

```yaml
data:
  logs_folder: ...
```

Each run saves a copy of the active configuration as:

```text
config_used.yaml
```

This is useful for reproducing a particular experiment.

## Testing

To run testing using an existing checkpoint, set:

```yaml
run_config:
  run_mode: 'test'

test_config:
  best_epoch_id: 12
  run_id: 9
```

Then run:

```bash
cd experiments/train_test
python main.py
```

Make sure that the checkpoint file exists in the corresponding log directory.

## Method summary

The correction model learns a mapping from the original WRF forecast to a corrected forecast. In the main configuration, the neural network predicts a correction term that is added to the original WRF field.

The total loss combines several terms:

* error with respect to ERA5 reanalysis;
* error with respect to station observations;
* error with respect to scatterometer measurements;
* a mesoscale-preserving term that penalizes changes in fine-scale structures of the original WRF forecast.

The fine-scale-preserving term is based on deviations from a Gaussian-smoothed local mean. This encourages the corrected forecast to retain high-frequency structures from the original high-resolution WRF field while correcting large-scale biases toward more reliable reference data.

## Citation

If you use this code or method in your research, please cite:

```bibtex
@article{golikov2026neural,
  title   = {Neural network atmospheric bias correction on heterogeneous data with fine-scale dynamics preservation},
  author  = {Golikov, Viktor and Krinitskiy, Mikhail and Gavrikov, Alexander and Burnaev, Evgeny and Vanovskiy, Vladimir},
  journal = {Quarterly Journal of the Royal Meteorological Society},
  year    = {2026},
  pages   = {e70209},
  doi     = {10.1002/qj.70209},
  url     = {https://doi.org/10.1002/qj.70209}
}
```

## License

Please see the repository license file for usage terms.

## Contact

For questions about the paper, data availability, or implementation details, please contact:

**Viktor Golikov**
Skolkovo Institute of Science and Technology
Email: [V.Golikov@skoltech.ru](mailto:V.Golikov@skoltech.ru)
