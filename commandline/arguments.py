import argparse
import utils

parser  =  argparse.ArgumentParser()

parser.add_argument('--experiment_name', default = None, help = "Name of the experiment")
parser.add_argument('--data_dir', default = None, help = "Directory containing the dataset")
parser.add_argument('--output_dir', default = None, help = "Directory to save outputs")
# parser.add_argument('--restore_file', default = None, help = "Optional, name of the file in --model_dir containing weights to reload before training")  # 'best' or 'train'
parser.add_argument('--seed', default = 1337, type = int, help = "Random seed")
parser.add_argument('--backbone', default = 'resnet18', type = str,help = "The backbone to use for the model")
parser.add_argument('--cfg', default = None,type = str, help = "The config file for the experiment")
parser.add_argument('--use_cuda', action = 'store_true', help = "Use cuda for training")
parser.add_argument('--use_cpu', action = 'store_true', help = "Use cpu for training")
# add argument to do one pass to check if the code is working
parser.add_argument('--dry_run', action = 'store_true', help = "Do one pass to check if the code is working")
parser.add_argument('--plain_resnet', action = 'store_true', help = "use adaptive downsampling")
parser.add_argument('--config_debug', action = 'store_true', help = "Print the config and exit")
# parser.add_argument("opts", default = None, nargs = argparse.REMAINDER, help = "modify config options using the command-line")
