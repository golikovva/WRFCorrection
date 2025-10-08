import itertools
import yaml
import torch
import os
import os.path as osp
import sys
from pathlib import Path
sys.path.append(str(Path('../').resolve()))
from train_test.new_main import main  
from correction.config.cfg import Config
def set_nested_attr(cfg, key_path, value):
    """
    Set a nested attribute in the configuration using dot-separated path.
    """
    obj = cfg
    keys = key_path.split('.')
    for k in keys[:-1]:
        obj = getattr(obj, k)
    setattr(obj, keys[-1], value)

if __name__ == "__main__":
    flag=True
    runs_to_exclude = [3, 4, 5, 12, 13, 36]
    print(runs_to_exclude)
    top_log_dir = '/home/logs/BERTunet/'    
    for run_dir in Path(top_log_dir).glob('misc_*'):

        cfg_path = os.path.join(run_dir, 'config_used.yaml')
        print(cfg_path)
        # with open(cfg_path) as f:
        #     cfg = yaml.safe_load(f)
        run_id = int(run_dir.name.split('_')[-1])
        print(run_id)
        if run_id != 3 and flag:
            continue
        else:
            flag=False
        if run_id in runs_to_exclude:
            print('Skipping run no:', run_id)
            continue
        best_epoch_filename = next(iter(run_dir.glob('models/model*.pth'))).stem
        best_epoch_id = int(best_epoch_filename.split('_')[-1])
        print(f'Testing run {run_id}, model epoch {best_epoch_id}')
        run_cfg = Config.fromfile(cfg_path)
        run_cfg.run_config.run_mode = 'test'
        run_cfg.test_config.run_id = run_id
        run_cfg.test_config.best_epoch_id = best_epoch_id
        run_cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        main(run_cfg)
        # torch.cuda.empty_cache()