import logging
import os
import yaml
import time

from launch import launch_dlrm, launch_wnd, launch_dien

def create_config(config_file="sda.yaml"):
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('SDA')
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def run_optimization(config):
    model = config['model']
    if model == 'DLRM':
        launch_dlrm(config)
    elif model == 'WnD':
        launch_wnd(config)
    elif model == 'DIEN':
        launch_dien(config)
    else:
        raise RuntimeError(f'Model {model} is not supported')

def main():
    config = create_config()
    logger = logging.getLogger('SDA')

    dataset = config['dataset']
    model = config['model']
    hosts = config['cluster']

    run_optimization(config)

if __name__ == '__main__':
    main()
