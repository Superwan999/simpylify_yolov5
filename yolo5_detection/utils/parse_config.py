import argparse
import os
import random
import torch
import yaml
from collections import OrderedDict
from os import path as osp
from dist_util import get_dist_info, init_dist, master_only
from general import set_random_seed


def ordered_yaml():
    """
    Support OrderedDict for yaml.
    :return: (tuple) yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def yaml_load(f):
    if os.path.isfile(f):
        with open(f, 'r') as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm'],
                        default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    opt = yaml_load(args.cfg)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed. ', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    return opt
