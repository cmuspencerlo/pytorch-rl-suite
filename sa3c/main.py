#!/usr/bin/env python3

import argparse
import gym
from util.pipeline import Pipeline

import numpy as np
import torch
from util import *
from tensorboardX import SummaryWriter

import math, os
os.environ["OMP_NUM_THREADS"] = "1"

SYNC_GLOBAL_STEP = 10
GAMMA = 0.9
MAX_EP = 50
MAX_EP_STEP = 200

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch on A3C experiment.')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env_str', default='Pendulum-v0', type=str, help='open-ai gym environment')
    
    parser.add_argument('--max_w_episode', default=30, type=int, help='max episode per worker')
    parser.add_argument('--max_w_episode_step', default=300, type=int, help='max step per episode')
    # parser.add_argument('--min_a', default=-1, type=float, help='min value for action')
    # parser.add_argument('--max_a', default=1, type=float, help='max value for action')
    
    args = parser.parse_args()
    
    pipeline = Pipeline(args)
    pipeline.train()
