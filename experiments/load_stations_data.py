import pandas as pd
from correction.config.config import cfg
from correction.validation.tasks import (
    get_measurements
)
borey_home = cfg.GLOBAL.BASE_DIR


start_date = '2022-09-07T21:00:00'
end_date = '2023-08-10T00:00:00'

meteostations = pd.read_csv(f'{borey_home}/metadata/meteostations.csv', comment='#')
for _, station in meteostations.iterrows():
    get_measurements_op = get_measurements(
        station, start_date, end_date, mnt_path=f'{borey_home}/mnt'
    )
    #
    # last_task = send_measurements(
    #     measurements_file=get_measurements_op,
    #     station=station,
    # )
