#!/usr/bin/env python3

import gym
import numpy as np
from util.util import *
from util.shared_adam import SharedAdam
from model.model import SimpleNet
from meta_architecture.A3C import Worker
import torch.multiprocessing as mp

class Pipeline(object):
    def __init__(self, args):
        self.args = args

    def train(self):
        env = gym.make(self.args.env_str) 
        N_S = env.observation_space.shape[0]
        N_A = env.action_space.shape[0]    
        global_net = SimpleNet(N_S, N_A)
        global_net.share_memory()   
        global_opt = SharedAdam(global_net.parameters(), lr=0.001)      # To be enrich.

        global_ep_step, global_ep_r, ret_pool = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
        workers = [Worker(self.args, global_net, global_opt, global_ep_step, ret_pool, i) for i in range(mp.cpu_count())]
        [w.start() for w in workers]
        [w.join() for w in workers]

        display_reward(ret_pool)
