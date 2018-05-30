#!/usr/bin/env python3

import os
import torch
from torch.autograd import Variable

CUDA_FLAG = torch.cuda.is_available()
LONG = torch.cuda.LongTensor if CUDA_FLAG else torch.LongTensor
FLOAT = torch.cuda.FloatTensor if CUDA_FLAG else torch.FloatTensor
DOUBLE = torch.cuda.DoubleTensor if CUDA_FLAG else torch.DoubleTensor

# Print functions
def print_red(s): print("\033[91m {}\033[00m" .format(s))
def print_green(s): print("\033[92m {}\033[00m" .format(s))
def print_yellow(s): print("\033[93m {}\033[00m" .format(s))
def print_lightpurple(s): print("\033[94m {}\033[00m" .format(s))
def print_purple(s): print("\033[95m {}\033[00m" .format(s))
def print_cyan(s): print("\033[96m {}\033[00m" .format(s))
def print_lightgray(s): print("\033[97m {}\033[00m" .format(s))
def print_black(s): print("\033[98m {}\033[00m" .format(s))

# Transform data format
def to_numpy(var):
    return var.cpu().data.numpy() if CUDA_FLAG else var.data.numpy()

# Assume tensor input
def to_var(tensor, volatile=False, requires_grad=False):
    tensor_type = tensor.type()
    if 'Float' in tensor_type:
        return Variable(tensor, volatile=volatile, requires_grad=requires_grad).type(FLOAT)
    if 'Double' in tensor_type:
        return Variable(tensor, volatile=volatile, requires_grad=requires_grad).type(DOUBLE)
    return Variable(tensor, volatile=volatile, requires_grad=requires_grad).type(LONG)

# def get_output_folder(parent_dir, env_name):
#     """Return save folder.
#
#     Assumes folders in the parent_dir have suffix -run{run
#     number}. Finds the highest run number and sets the output folder
#     to that number + 1. This is just convenient so that if you run the
#     same script multiple times tensorboard can plot all of the results
#     on the same plots with different names.
#
#     Parameters
#     ----------
#     parent_dir: str
#       Path of the directory containing all experiment runs.
#
#     Returns
#     -------
#     parent_dir/run_dir
#       Path to this run's save directory.
#     """
#     os.makedirs(parent_dir, exist_ok=True)
#     experiment_id = 0
#     for folder_name in os.listdir(parent_dir):
#         if not os.path.isdir(os.path.join(parent_dir, folder_name)):
#             continue
#         try:
#             folder_name = int(folder_name.split('-run')[-1])
#             if folder_name > experiment_id:
#                 experiment_id = folder_name
#         except:
#             pass
#     experiment_id += 1
#
#     parent_dir = os.path.join(parent_dir, env_name)
#     parent_dir = parent_dir + '-run{}'.format(experiment_id)
#     os.makedirs(parent_dir, exist_ok=True)
#     return parent_dir


