import os
from commandline.arguments import parser
from configs import setup_cfg
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import utils
from model import network
from training import train
import logging
from dataset import data


logging.basicConfig(level=logging.NOTSET)
HANDLE = "CAD"
logger = logging.getLogger(HANDLE)


def main():
    logger.info("Reading arguments...")
    args = parser.parse_args()
    cfg = setup_cfg(args)
    logger.info("Setting seed...")
    utils.set_random_seed(cfg.SEED)

    if (cfg.CONFIG_DEBUG):
        print(cfg)
    else:
        # model = network._resnet('resnet50', "Bottleneck", [3, 4, 6, 3], False, True)
        model = network._resnet(cfg)
        train_loader, val_loader = data.get_data_loaders(cfg)
        train(model, train_loader, val_loader, cfg)

if __name__ == '__main__':
    main()