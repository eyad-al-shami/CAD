from commandline.arguments import parser
from configs import setup_cfg
import utils
from model import network
import training
import plain_resnet_training
import logging
from dataset import data
import wandb
import torch

# Ignore warnings of PIL
logging.getLogger("PIL").setLevel(logging.WARNING)


logging.basicConfig(level=logging.NOTSET)
HANDLE = "CAD"
logger = logging.getLogger(HANDLE)


def main():
    logger.info("Reading arguments...")
    args = parser.parse_args()
    cfg = setup_cfg(args)
    logger.info("Setting seed...")
    utils.set_random_seed(cfg.SEED)

    # wandb.init(project=cfg.LOGGING.PROJECT, entity="mengyuanliu")
    wandb.init(project=cfg.LOGGING.PROJECT, entity="sawt", name=cfg.LOGGING.EXPERIMENT_NAME,  config=cfg)

    if (cfg.CONFIG_DEBUG):
        print(cfg)
    else:
        train_loader, val_loader = data.get_data_loaders(cfg)
        if cfg.MODEL.AD:
            logger.info("++++++++++++ Training Adaptive Sampling-based resnet ++++++++++++")
            model = network._resnet(cfg)
            training.train(model, train_loader, val_loader, wandb, cfg)
        else:
            logger.info("++++++++++++ Training plain resnet ++++++++++++")
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
            plain_resnet_training.train(model, train_loader, val_loader, wandb, cfg)

if __name__ == '__main__':
    main()