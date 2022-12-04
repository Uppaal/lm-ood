import os
import sys
import pickle
import logging
import socket
import random
import datetime
import numpy as np
from typing import List

# PyTorch
import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers import BertModel, BertConfig, BertTokenizer

# # Tensorflow
# import tensorflow as tf
# from transformers import T5Tokenizer, T5Config, TFT5ForConditionalGeneration
# tf.get_logger().setLevel(logging.ERROR)
# tf.config.optimizer.set_jit(True) # For speedups

machine = 'galaxy' if 'galaxy' in socket.gethostname() else 'local'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if machine == 'galaxy':
    root_path = os.path.join('/afs/cs.wisc.edu/u/u/p/uppaal/Code')
    sys.path.extend([(root_path + source_path) for source_path in os.listdir(root_path) if not (root_path + source_path) in sys.path])

class Map(object): pass


class HF_Model():
    def __init__(self, model_class=None, model_path=None, max_sequence_length=2048, global_batch_size=None):
        self.model_class = model_class

        self.model_path = model_path
        if self.model_path is None:
            # TODO: Based on model type, find path
            if model_class == 'BERT':
                self.model_path = '/u/u/p/uppaal/huggingface_models/bert_base_uncased'

        self.max_sequence_length = max_sequence_length

        self.train_vars = Map()
        self.train_vars.global_batch_size = global_batch_size

        self.load_tokenizer_and_model()

    def load_tokenizer_and_model(self):
        if self.model_class == 'BERT':
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self.model_config = BertConfig()
            self.model = BertModel.from_pretrained(self.model_path, config=self.model_config)
        else:
            #TODO: For other classes
            logger.warning(f'No support for {self.model_class}.')
        logger.info(f'Loaded model {self.model_class} from {self.model_path}.')

    def load_model_from_checkpoint(self, checkpoint_name):
        if self.model_class == 'BERT':
            self.model =  BertModel.from_pretrained(os.path.join(self.model_path, checkpoint_name))


    def train(self):
        def save_model_to_disk(self, model_name, epoch):
            self.model.save_pretrained(os.path.join(self.model_path, model_name))
            pickle.dump(self.train_vars.optimizer.get_weights(), open(os.path.join(self.model_path, model_name, 'optimizer.pkl'), 'wb'))
            logger.info(f'Epoch {epoch}: Saved model and optimizer state to {os.path.join(self.model_path, model_name)}.')



HF_Model(model_class='BERT', model_path='')



