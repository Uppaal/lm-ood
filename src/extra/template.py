# # Tensorflow
# import tensorflow as tf
# from transformers import T5Tokenizer, T5Config, TFT5ForConditionalGeneration
# tf.get_logger().setLevel(logging.ERROR)
# tf.config.optimizer.set_jit(True) # For speedups

import os
import sys
import time
import torch
import pickle
import random
import socket
import logging
import datetime
import collections
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, DatasetDict

torch.cuda.empty_cache() # Save memory
class Map(object): pass
machine = 'galaxy' if 'galaxy' in socket.gethostname() else 'local'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if machine == 'local':
    MODEL_DIR = '/models'
    DATA_DIR = '/data'
    root_path = 'C:/Users/rheey/Desktop/Academic Drive/Spring 22/Research/Code'
elif machine == 'galaxy':
    MODEL_DIR = '/afs/cs.wisc.edu/u/u/p/uppaal/models'
    DATA_DIR = '/afs/cs.wisc.edu/u/u/p/uppaal/data'
    root_path = '/afs/cs.wisc.edu/u/u/p/uppaal/Code'

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if machine =='galaxy':
    logger.info(f'Using GPUS: {os.environ["CUDA_VISIBLE_DEVICES"]}')
sys.path.extend([(root_path + source_path) for source_path in os.listdir(root_path) if not (root_path + source_path) in sys.path])

from pynvml import *
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

############################################################################

# Check if torch is using GPUs
torch.cuda.is_available() # True
torch.cuda.current_device() # 0
torch.cuda.device_count() # 1
torch.cuda.get_device_name(0) # GPU model name

############################################################################

# Setting Random Seeds
import random
random.seed(0)

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.random.set_seed(0)

import torch
torch.manual_seed(0)
torch.random.manual_seed(0)
torch.random.seed()

############################################################################

# HuggingFace Datasets
# Ref: https://pytorch.org/hub/huggingface_pytorch-transformers/

from datasets import load_dataset
dataset = load_dataset('LOADING_SCRIPT', cache_dir="PATH/TO/MY/CACHE/DIR")

############################################################################

# Torch Hub

import torch
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')    # Download vocabulary from S3 and cache.
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', './test/bert_saved_model/')  # E.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`

# Get list of pretrained models
torch.hub.list(github='huggingface/pytorch-transformers') # Saved to /tmp/3393/cache/torch/hub/master.zip

############################################################################

# Note GPU utilization

from pynvml import *
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")
def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
result = trainer.train() # Huggingface trainer
print_summary(result)

############################################################################

# For inference
model.eval() # evaluation mode turns off dropout regularization which is used in training
with torch.no_grad():  # tells PyTorch not to construct the compute graph during this forward pass (since we won’t be running backprop here)–this just reduces memory consumption and speeds things up a little.
    outputs = model(inputs)

hidden_states = outputs.hidden_states
len(hidden_states) # Number of layers - initial embeddings + 12 BERT layers
hidden_states[i] # batch_size x sequence length x hidden dim

# Getting sentence embeddings using SentenceBERT/ SentenceRoberta
#https://github.com/UKPLab/sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('roberta-base')
sentence_embeddings = model.encode(['Hi', 'Hello'])

