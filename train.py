from utils.logger import Logger
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
parser.add_argument('--seed', type=int, default=0,
                    help='Set the random seed for reprodicibility')
parser.add_argument('--disable-cuda', action='store_true', default=False,
                    help='Flag to DISABLE CUDA (which is ENABLED by default)')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Flag to debug mode')
parser.add_argument('--gpu', type=int, default=0,
                    help='Select which GPU to use (e.g., 0, 1, 2, or 3)')

args = parser.parse_args()

# if not args.disable_cuda and torch.cuda.is_available():
#     use_cuda = True
#     device = torch.device('cuda:{}'.format(args.gpu))
# else:
#     device = torch.device('cpu')
#
with open(args.config) as f:
    config = yaml.load(f)
logger = Logger(config)