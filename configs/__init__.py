from .defaults import _C as cfg_default
import utils
import torch
import logging
HANDLE = "CAD"
logger = logging.getLogger(HANDLE)

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return cfg_default.clone()

def merge_args_cfg(cfg, args):
    # TODO logging must always be enabled, use tensorboard by default
    if (args.experiment_name):
        cfg.LOGGING.EXPERIMENT_NAME = args.experiment_name
    else:
        if (cfg.LOGGING.EXPERIMENT_NAME == ""):
            logger.warning("No experiment name specified, using default")
            cfg.LOGGING.EXPERIMENT_NAME = cfg.LOGGING.PROJECT + " " + utils.get_readable_date_time()

    if args.data_dir:
        cfg.DATA.DATA_PATH = args.data_dir

    cfg.OUTPUT_DIR = args.output_dir if args.output_dir else cfg.LOGGING.EXPERIMENT_NAME
    # remove andy characters from the output directory name that may cause problems
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace(":", "_")
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace(" ", "_")
    invalid = '<>:"/\|?*'
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.translate({ord(c): None for c in invalid})



    # if args.restore_file:
    #     cfg.RESUME = args.restore_file

    if args.seed:
        cfg.SEED = args.seed

    if  args.use_cuda:
        # check if cuda is available
        if torch.cuda.is_available():
          cfg.USE_CUDA = args.use_cuda
          cfg.DEVICE = "cuda"
        else:
          logging.warning("The specified device (CUDA) is not available, using CPU instead")
          cfg.USE_CUDA = False
          cfg.DEVICE = "cpu"
    
    if (args.config_debug):
        cfg.CONFIG_DEBUG = True

    if args.dry_run:
        cfg.DRY_RUN = True

    if args.use_cpu:
        if (torch.cuda.is_available()):
            logging.warning("CUDA is available, but CPU was specified, using CPU anyway")
        cfg.USE_CUDA = False
        cfg.DEVICE = "cpu"

    
    # cfg.merge_from_list(args.opts)

def setup_cfg(args):
    cfg = get_cfg_defaults()
    
    # 1. From experiment config file
    if args.cfg:
        logger.info("Loading config from {}".format(args.cfg))
        # print("Loading config from {}".format(args.cfg))
        cfg.merge_from_file(args.cfg)

    # 2. From input arguments
    merge_args_cfg(cfg, args)
    # 4. From optional input arguments
    # cfg.merge_from_list(args.opts)
    # clean_cfg(cfg, args.trainer)
    
    cfg.freeze()
    return cfg
