import gc
import torch
import socket
import logging

# Set loggers and objects
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Util for GPU utilization
from pynvml import *
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

# Save memory
gc.collect()
torch.cuda.empty_cache()

class Map(object): pass
paths = Map()
run_configs = Map()

# Machine configs
run_configs.machine = 'galaxy' if 'galaxy' in socket.gethostname() else 'local'
run_configs.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Directory configs
if run_configs.machine == 'local':
    paths.root_path = 'C:/Users/rheey/Desktop/Academic Drive/Research/Project - Pretrained models for OOD Detection/Code/ood-in-text/src'
    paths.MODEL_DIR = f'{paths.root_path}/../models'
    paths.MODEL_DIR_local = paths.MODEL_DIR
    paths.DATA_DIR = f'{paths.root_path}/../data'
    paths.output_plot_dir = f'{paths.root_path}/../plots'
    paths.temp_output_dir = f'{paths.root_path}/../temp_outputs'
    # paths.MODEL_DIR = '../models'
    # paths.DATA_DIR = '../data'
    # paths.output_plot_dir = '../plots'
    # paths.temp_output_dir = '../temp_outputs'
elif run_configs.machine == 'galaxy':
    paths.root_path = '/afs/cs.wisc.edu/u/u/p/uppaal/Code/ood-in-text/src'
    paths.MODEL_DIR = '/nobackup/uppaal/checkpoints'
    paths.MODEL_DIR_local = '/afs/cs.wisc.edu/u/u/p/uppaal/Models'
    paths.DATA_DIR = '/afs/cs.wisc.edu/u/u/p/uppaal/Data'
    paths.output_plot_dir = f'{paths.root_path}/../plots'
    paths.temp_output_dir = f'{paths.root_path}/../temp_outputs'
    # paths.output_plot_dir = '../plots'
    # paths.temp_output_dir = '../temp_outputs'

if run_configs.machine =='galaxy':
    logger.info(f'Using GPUS: {os.environ["CUDA_VISIBLE_DEVICES"]}')

# Run configs
run_configs.debug_mode = True
run_configs.set_seed = True
run_configs.max_seq_len = 256
