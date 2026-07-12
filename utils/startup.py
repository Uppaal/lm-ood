import gc
import os

import nvsmi
import socket
import random
import logging
import numpy as np
from pynvml import *
from utils.configs import Config


# Set logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Map(object): pass


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


exp_configs = Config()
import torch
if exp_configs.set_seed:
    set_seed(exp_configs)
exp_configs.values()

logger.info(f'Host: {socket.gethostname()}')
free_gpus = [x for x in nvsmi.get_available_gpus()]
logger.info(f'Free GPUs: {[x.id for x in free_gpus]}')
if torch.cuda.is_available():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    logger.info(f'GPU in use: {gpu_name} (ID: {visible_devices})')
else:
    logger.info('GPU in use: CPU')


# Save memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# Util for GPU utilization
def print_gpu_utilization():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])

    # input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_ids = [list(f["input_ids"].numpy()) + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    outputs = {
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "labels": labels,
    }
    return outputs
