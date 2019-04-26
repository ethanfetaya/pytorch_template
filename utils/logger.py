from datetime import datetime
from ruamel.yaml import YAML
import os, shutil
import sys
yaml = YAML()

def format_time():
    t = datetime.now()
    s = t.strftime('%Y-%m-%d_%H:%M:%S.%f')
    return s[:-3]

class Logger(object):
    def __init__(self,config):
        log_config = config['logging']
        debug = config['debug']
        base_folder = log_config['save_path']
        if not debug:
            save_folder = os.path.join(base_folder, log_config['exp_name'] + '__' + format_time())
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            else:
                raise RuntimeError('Save folder already exists.')
        else:
            save_folder = os.path.join(base_folder, 'debug')
            if os.path.isdir(save_folder):
                shutil.rmtree(save_folder)
            os.makedirs(save_folder)
        config_save = os.path.join(save_folder,'config.yaml')
        with open(config_save, 'w') as f:
            yaml.dump(config, f)
        
