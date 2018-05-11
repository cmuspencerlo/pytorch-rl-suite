#!/usr/bin/env python3

import os
import torch
from torch.autograd import Variable

# CUDA_FLAG = torch.cuda.is_available()
CUDA_FLAG = False
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

def to_var(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

def record(global_ep_step, ret_pool, epoch_reward, id):
    with global_ep_step.get_lock():
        global_ep_step.value += 1
    ret_pool.put(epoch_reward)
    print(
        id,
        'Epoch:', global_ep_step.value,
        '| Ep_r: %.0f' % epoch_reward,
    )

def display_reward(ret_pool):
    ret_plt = []
    while not ret_pool.empty():
        r = ret_pool.get()
        if not ret_plt:
            ret_plt.append(r)
        else:
            prev_r = ret_plt[-1]
            ret_plt.append(prev_r * 0.99 + r * 0.01)
    
    import matplotlib.pyplot as plt
    plt.plot(ret_plt)
    plt.show()
