# my_project/config.py
from yacs.config import CfgNode as CN

###########################
# Config definition
###########################

_C = CN()
_C.USE_CUDA = True
_C.DEVICE = "cuda" if _C.USE_CUDA else "cpu"
# Set seed to negative value to randomize everything
# Set seed to positive value to use a fixed seed
_C.SEED = 1337
# Print detailed information
# E.g. trainer, dataset, and backbone
_C.VERBOSE = True
_C.DRY_RUN = False
_C.CONFIG_DEBUG = False
_C.OUTPUT_DIR = "saved_models"

###########################
# System Hardware
###########################

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.NUM_WORKERS = 4

###########################
# Training
###########################

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 60

###########################
# Logging
###########################

_C.LOGGING = CN()
_C.LOGGING.EXPERIMENT_NAME = ""
_C.LOGGING.LOG_DIR = "logs"
_C.LOGGING.PROJECT = "content-adaptive downsampling"


###########################
# Models
###########################
_C.MODEL = CN()
_C.MODEL.BACKBONE = "resnet50"
_C.MODEL.PRETRAINED = True
_C.MODEL.BLOCK = "Bottleneck"
_C.MODEL.LAYERS = [3, 4, 6, 3]
_C.MODEL.LR = 0.001
_C.MODEL.AD = True
_C.MODEL.MASK_LOW_RES_ACTIVE = 0.3

###########################
# Dataset
###########################

_C.DATA = CN()
_C.DATA.BATCH_SIZE = 32
_C.DATA.DATA_PATH = ''
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.TRAIN_INTERPOLATION = 'bicubic'


# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`