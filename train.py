from utils.logger import Logger
from utils.utils import get_model, get_data, get_optimizer
import argparse
import numpy as np
import torch

# import torch.optim as optim
# import torch.nn.functional as F

from ruamel.yaml import YAML
yaml=YAML(typ='safe')

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str,
                    help='Path to YAML configuration file.')
parser.add_argument('--disable-cuda', action='store_true', default=False,
                    help='Flag to DISABLE CUDA (which is ENABLED by default)')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Flag to debug mode')
parser.add_argument('--gpu', type=int, default=0,
                    help='Select which GPU to use (e.g., 0, 1, 2, or 3)')

args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f)
config['config_file'] = args.config.replace('/','.').split('.')[-2]

seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

n_epochs = config['optimization']['n_epochs']

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(args.gpu))
else:
    device = torch.device('cpu')

logger = Logger(config)
model = get_model(config['model'])
optim = get_optimizer(model.parameters(),config['optimization'])
train_loader, valid_loader, test_loader = get_data(config['data'])

## Train
for i in range(n_epochs):
    for data, label in train_loader:
        break


