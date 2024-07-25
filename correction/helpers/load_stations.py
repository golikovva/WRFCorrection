import io
import numpy as np
import os
import pandas as pd
import pendulum as pdl
import re
import requests
import time
import wget

from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm


def load_station_data(station, meteo_period, attempts=2, attempt_delay=0.3):
    page_url = f'https://rp5.ru/Weather_archive_in_{station.ref}'
    response = requests.get(page_url)
    phpsessid = response.cookies['PHPSESSID']
    cookies = dict(
        PHPSESSID=phpsessid,
        located='1',
        extreme_open='false',
        iru='7491',
        itr='7491',
        full_table='1',
        zoom='9',
        tab_synop='2',
        tab_metar='1',
        format='csv',
        f_enc='utf',
        lang='en',
    )

    headers = {
        'Accept': 'text/html, */*; q=0.01',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'ru,en-US;q=0.9,en;q=0.8',
        'Connection': 'keep-alive',
        'Content-Length': '99',
        'Content-Type': 'application/x-www-form-urlencoded',
        'DNT': '1',
        'Host': 'rp5.ru',
        'Origin': 'https://rp5.ru',
        'Referer': 'https://rp5.ru/',
        'sec-ch-ua': '"Not_A Brand";v="99", "Google Chrome";v="109", "Chromium";v="109"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
    }

    a_date1 = meteo_period.start.strftime('%d.%m.%Y')
    a_date2 = meteo_period.end.strftime('%d.%m.%Y')
    data = {
        'a_date1': a_date1,
        'a_date2': a_date2,
        'f_ed3': '1',
        'f_ed4': '1',
        'f_ed5': '1',
        'f_pe': '1',
        'f_pe1': '2',
        'lng_id': '1',
    }
    if station.type == 'synop':
        data['wmo_id'] = station.id
    elif station.type == 'metar':
        data['metar'] = station.id
    else:
        raise RuntimeError(f'Unexpected station type: {station.type}.')

    url = None
    iterator = tqdm(range(attempts), desc='attempt')
    for _ in iterator:
        print(f'https://rp5.ru/responses/reFile{station.type.capitalize()}.php')
        response = requests.post(
            f'https://rp5.ru/responses/reFile{station.type.capitalize()}.php',
            cookies=cookies, headers=headers, data=data, verify=False
        )
        print(response)
        # response = requests.post(url, headers=headers, data=myData, verify=False)
        match = re.search(r'<a href=(\S*) download>', response.text)
        if match is not None:
            url = match[1]
            break
        else:
            time.sleep(attempt_delay)
    if url is None:
        print('Was unable to obtain url from website. Trying to load file manually.')
        print(os.getcwd())
        proj_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        print(proj_dir)
        tmp_file = os.path.join(proj_dir, 'stations','ULAA.01.01.2016.10.08.2023.1.0.0.en.utf8.00000000.csv',
                                'ULAA.01.01.2016.10.08.2023.1.0.0.en.utf8.00000000.csv')
        station_data = pd.read_csv(tmp_file, sep=';', comment='#', encoding='latin-1', skiprows=6)
                                # f'station_{station.tslist}_{a_date1}_{a_date2}.csv')
        # raise RuntimeError('Was unable to obtain url from website.')
    else:
        tmp_file = wget.download(url)
        station_data = pd.read_csv(tmp_file, sep=';', comment='#')
        os.remove(tmp_file)
    if station.type == 'synop':
        station_data = station_data.reset_index()
        station_data.columns = station_data.columns.tolist()[1:] + ['none']
        station_data = station_data.drop('none', axis=1)
    return station_data


def manual_load_station_data(tmp_file, station_type='metar'):
    station_data = pd.read_csv(tmp_file, sep=';', comment='#')
    os.remove(tmp_file)
    if station_type == 'synop':
        station_data = station_data.reset_index()
        station_data.columns = station_data.columns.tolist()[1:] + ['none']
        station_data = station_data.drop('none', axis=1)
    return station_data


def process_station_data(station_data, station):
    station_data['timestamp'] = station_data[f'Local time {station.article} {station.title}'].apply(
        lambda s: pdl.from_format(f'{s} {station.tz}', 'DD.MM.YYYY HH:mm Z').in_tz('UTC').timestamp()
    )
    station_data['TC'] = station_data['T'] + 273.15
    station_data['PSFC'] = station_data['Po' if station.type == 'synop' else 'P0']
    station_data['WSPD10'] = station_data['Ff']
    print(station_data.columns)
    station_data = station_data[['timestamp', 'TC', 'PSFC', 'WSPD10']]
    station_data = station_data.sort_values(by='timestamp')
    station_data = station_data.interpolate(limit_direction='both')
    return station_data


def process_station_tslist(station_ts, run_period_start):
    process_station_tslist.zero_celsius = 273.15
    process_station_tslist.pa_to_mmhg = 0.0075006157584566

    station_ts['timestamp'] = station_ts['ts_hour'].apply(
        lambda hour: (run_period_start + pdl.duration(hours=hour)).timestamp()
    )
    station_ts['TK'] = station_ts['t'].values  # - process_station_tslist.zero_celsius
    station_ts['PSFC'] = station_ts['psfc'].values * process_station_tslist.pa_to_mmhg
    station_ts['WSPD10'] = np.sqrt(station_ts['u'].values ** 2 + station_ts['v'].values ** 2)
    station_ts['U10'] = station_ts['u'].values
    station_ts['V10'] = station_ts['v'].values

    station_ts = station_ts[['timestamp', 'TC', 'PSFC', 'WSPD10', 'U10', 'V10']]
    return station_ts


def interpolate_to_period(df, meteo_period):
    timestamp_grid = np.array([
        dt.timestamp() for dt in meteo_period.range(unit='hours', amount=1)
    ])
    result = pd.DataFrame()
    for colname, column in df.items():
        result[colname] = np.interp(
            timestamp_grid, df['timestamp'].values, column.values
        )
    return result


def get_station_data(station, meteo_period):
    station_data = load_station_data(station, meteo_period)
    station_data = process_station_data(station_data, station)
    station_data = interpolate_to_period(station_data, meteo_period)
    return station_data


def get_station_tslists(station, meteo_period, mnt_path):
    get_station_tslists.ts_colnames = [
        'id', 'ts_hour', 'id_tsloc', 'ix', 'iy', 't', 'q', 'u', 'v', 'psfc',
        'glw', 'gsw', 'hfx', 'lh', 'tsk', 'tslb(1)', 'rainc', 'rainnc', 'clw',
    ]
    get_station_tslists.use_ts_cols = [
        'ts_hour', 't', 'u', 'v', 'psfc',
    ]

    mnt_path = Path(mnt_path)
    runs = mnt_path.glob('run_*')
    result = {}
    for run in runs:
        run_start_dt = pdl.parse(run.name[4:]).in_tz('UTC')
        run_period = pdl.period(
            run_start_dt - pdl.duration(days=1),
            run_start_dt + pdl.duration(days=3)
        )
        intersection = pdl.period(
            max(meteo_period.start, run_period.start),
            min(meteo_period.end, run_period.end),
        )
        if intersection.total_hours() > 0:
            ts_path = run / f'{station.tslist}.d01.TS'
            station_ts = pd.read_csv(
                ts_path, delim_whitespace=True, names=get_station_tslists.ts_colnames,
                skiprows=1, usecols=get_station_tslists.use_ts_cols
            )
            station_ts = process_station_tslist(station_ts, run_period.start)
            station_ts = interpolate_to_period(station_ts, meteo_period)
            result[f'run {run_start_dt}'] = station_ts
    return result


def plot_station_label(measurements, label, title):
    fig = plt.figure(figsize=(8, 5))

    for name in sorted(measurements):
        data = measurements[name]
        data_xs = data['timestamp'].apply(lambda ts: pdl.from_timestamp(ts))
        plt.plot(data_xs, data[label], label=name)

    plt.xlabel('Datetime')
    plt.ylabel(label)
    plt.grid()
    plt.legend()
    plt.title(title)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight', format='png')
    plt.close()

    buf.seek(0)
    return buf
