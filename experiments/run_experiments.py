import itertools
import copy
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
    # Path to your base configuration file
    num_trials = 3
    base_config_path = osp.join(osp.dirname(__file__), os.pardir, 'correction', 'config', 'config.yaml')
    
    # Define parameter grid (dot-separated paths for nested attributes)
    param_grid = {
        'betas.beta1': [1],
        'betas.beta2': [10],
        'betas.beta3_t2': [0],
        'betas.beta4': [0],
        'model_type': ['BERTunet_raw', ]
    }
    
    # Generate all combinations of parameters
    keys = param_grid.keys()
    value_combinations = list(itertools.product(*param_grid.values()))
    experiments = [[dict(zip(keys, combo)) for combo in value_combinations] for _ in range(num_trials)]
    
    print(f"Running {len(experiments)} sets of experiments...")
    for tryal_i, tryal in enumerate(experiments):
        # tryal.pop(0)
        for i, params in enumerate(tryal):
            print(params)
            params['betas.beta3_w10'] = 2 * params['betas.beta3_t2']
            print(f"\n=== Tryal no {tryal_i+1} ===\n=== Experiment {i+1}/{len(tryal)}: {params} ===")
            
            # try:
            # Load fresh base config
            cfg = Config.fromfile(base_config_path)
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
            # Apply modifications
            for key_path, value in params.items():
                set_nested_attr(cfg, key_path, value)
            
            # Run training/testing
            main(cfg)
                
            # except Exception as e:
            #     print(f"Experiment failed: {str(e)}")
            
            # finally:
            #     # Clear GPU memory
            #     torch.cuda.empty_cache()