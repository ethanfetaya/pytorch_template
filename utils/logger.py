from datetime import datetime
from ruamel.yaml import YAML
from tensorboardX import SummaryWriter
import os, shutil
import sys
import numpy as np
yaml = YAML()
from utils.utils import save_models

def format_time():
    t = datetime.now()
    s = t.strftime('%Y-%m-%d_%H:%M:%S.%f')
    return s[:-3]


class Logger(object):
    def __init__(self,config):
        log_config = config['logging']
        debug = config['debug']
        base_folder = log_config['base_dir']
        if not debug:
            self.save_folder = os.path.join(base_folder,config['config_file'], log_config['exp_name'],format_time())
            if not os.path.isdir(self.save_folder):
                os.makedirs(self.save_folder)
            else:
                raise RuntimeError('Save folder already exists.')
        else:
            self.save_folder = os.path.join(base_folder, 'debug')
            if os.path.isdir(self.save_folder):
                shutil.rmtree(self.save_folder)
            os.makedirs(self.save_folder)
        config_save = os.path.join(self.save_folder,'config.yaml')
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            config['git_commit'] = sha
        except:
            pass
        with open(config_save, 'w') as f:
            yaml.dump(config, f)
        self.write = SummaryWriter(log_dir=self.save_folder)
        self.best_loss = np.inf

    def save_best(self,models,loss,optimizers=None):
        if loss<self.best_loss:
            best_loss = loss
            save_models(models,self.save_folder,prefix='best')
        save_models(models, self.save_folder, prefix='last')
        if optimizer is not None:
            save_models(optimizers, self.save_folder, prefix='last_optimizer')
        return
