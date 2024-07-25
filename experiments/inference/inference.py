import os
import time
import argparse
import xarray as xr
import numpy as np
import torch
from yaml import load, SafeLoader
import sys

sys.path.insert(0, '../../')
from correction.data.my_dataloader import WRFDataset
from correction.data.scaler import StandardScaler
from correction.models.build_module import build_inference_correction_model


def run_correction(parameters):
    start = time.time()
    ds = xr.open_dataset(parameters['input']).copy(deep=True)
    print('Loading model...')
    model = build_inference_correction_model(parameters)

    if parameters['output'] is None:
        output_file = parameters['input'][:-3] + '_corrected.nc'
    elif parameters['output'] == parameters['input']:
        # os.remove(parameters['input'])
        output_file = parameters['input'][:-3] + '_corrected.nc'
    else:
        output_file = parameters['output']
    print('Applying model...')
    prepare_input(ds, parameters)
    exit()
    ds[parameters['salinity_var']].data = apply_model(ds, model, parameters)
    os.remove(output_file) if os.path.exists(output_file) else None
    ds.to_netcdf(output_file)
    print(f'Time spent to apply correction: {round(time.time() - start, 2)} s')
    return parameters['output']


def prepare_input(dataset, parameters):
    date = dataset['XTIME'].data
    day_normalized = (date.astype('datetime64[D]') - date.astype('datetime64[Y]')).astype(int) / 365
    hour_normalized = (date.astype('datetime64[h]') - date.astype('datetime64[D]')).astype(int) / 24

    input_data = WRFDataset.load_file_vars(parameters['input'], parameters['input_vars'])
    lon = WRFDataset.load_file_vars(parameters['input'], [parameters['longitude_var']])
    lat = WRFDataset.load_file_vars(parameters['input'], [parameters['latitude_var']])

    d_encoded = WRFDataset.get_time_encoding(day_normalized, lon[0, 0], lat[0, 0], periodic_law=np.sin)[:, None]
    h_encoded = WRFDataset.get_time_encoding(hour_normalized, lon[0, 0], lat[0, 0], periodic_law=np.cos)[:, None]
    land_mask = WRFDataset.create_landmask_from_sample(parameters['input'])

    land_mask = np.broadcast_to(land_mask, d_encoded.shape).copy()  # 24 1 210 280
    print(d_encoded.shape, lat.shape, lon.shape, h_encoded.shape, land_mask.shape, input_data.shape, lat.shape,
          lon.shape)
    model_input = torch.from_numpy(np.concatenate([input_data,
                                                   lat, lon,
                                                   land_mask,
                                                   d_encoded, h_encoded], axis=1)).float()  # 24 9 210 280
    print(model_input.shape)
    model_input = model_input.to(parameters['device'])
    return model_input


def apply_model(dataset, model, parameters):
    model_input = prepare_input(dataset, parameters)
    means_dict = torch.load(parameters['means_path'])
    stds_dict = torch.load(parameters['stds_path'])
    if "unet" in parameters['model_type']:
        time_keys = ['day', 'hour']
        landmask_key = ['LANDMASK']
        wrf_keys = ['u10', 'v10', 'T2'] + ['XLAT', 'XLONG'] + landmask_key + time_keys
        scaler = StandardScaler()
        scaler.apply_scaler_channel_params(torch.tensor([means_dict[x] for x in wrf_keys]).float(),
                                           torch.tensor([stds_dict[x] for x in wrf_keys]).float())
        scaler.to(parameters['device'])
        with torch.no_grad():

            model_input = scaler.transform(model_input, dims=1)
            out = model(model_input.view(-1, 4, 9, 210, 280))
            out = scaler.inverse_transform(out, means=scaler.means[0], stds=scaler.stddevs[0], dims=1)
            out = out.flatten(0, 1).cpu().numpy()
        return out

    else:
        raise NotImplementedError


if __name__ == '__main__':
    """
    example inference command:
    python inference.py /home/glorys_op/glorys_24010200/GLORYS_2024-01-02_00_cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m.nc \
    config.yaml \
    -o /home/logs/spatiotemporal_baseline/GLORYS_2024-01-02_00_phy-so_corrected_1.nc

    """
    parser = argparse.ArgumentParser(
        prog='Glorys salinity correction',
        description='Correct glorys data from the input glorys data and correction field',
    )
    parser.add_argument(
        'input',
        type=str,
        help='path to the file to correct',
    )
    parser.add_argument(
        'namelist',
        type=str,
        help='path to the namelist file'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='path to the output file',
    )

    args_dict = vars(parser.parse_args())
    with open(args_dict['namelist'], 'r') as namelist_file:
        namelist = load(namelist_file, Loader=SafeLoader)
    parameters = {**args_dict, **namelist}

    saved_filename = run_correction(parameters)
