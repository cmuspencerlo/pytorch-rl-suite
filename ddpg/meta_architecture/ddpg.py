#!/usr/bin/env python3

# import sys
# sys.path.append('../')

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model.model import (Actor, Critic)
from util.memory import SequentialMemory
from util.random_process import OrnsteinUhlenbeckProcess
from util.util import *

# from ipdb import set_trace as debug

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, args, num_states, num_actions):

        
        #     self.seed(args.seed)

        self.num_states = num_states
        self.num_actions = num_actions
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        self.actor = Actor(self.num_states, self.num_actions, **net_cfg)
        self.actor_target = Actor(self.num_states, self.num_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.num_states, self.num_actions, **net_cfg)
        self.critic_target = Critic(self.num_states, self.num_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)
        if CUDA_FLAG: self.network_cuda()

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=num_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.k_tau = args.tau
        self.k_discount = args.discount
        self.k_depsilon = 1.0 / args.epsilon
        self.k_epsilon = 1.0
        self.curr_s = None # Most recent state
        self.curr_a = None # Most recent action
        self.min_a = args.min_a
        self.max_a = args.max_a

    def observe(self, step, new_reward, new_state, done, trainable_flag):
        self.memory.append(self.curr_s, self.curr_a, new_reward, done)
        self.curr_s = new_state
        if trainable_flag:
            self._update_policy()

    def _update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Update Critic network
        next_q_batch = self.critic_target([
            to_var(next_state_batch, volatile=True), 
            self.actor_target(to_var(next_state_batch, volatile=True))
        ])

        # Magic
        next_q_batch.volatile=False

        target_q_batch = to_var(reward_batch) + \
            self.k_discount * to_var(terminal_batch) * next_q_batch
        q_batch = self.critic([to_var(state_batch), to_var(action_batch)])

        self.critic_optim.zero_grad()
        q_value_loss = criterion(q_batch, target_q_batch)
        q_value_loss.backward()
        self.critic_optim.step()

        # Update Actor network
        # Magic: what if critic stop gradient update here?
        self.actor_optim.zero_grad()
        policy_loss = -self.critic([
            to_var(state_batch),
            self.actor(to_var(state_batch))]).mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor, self.actor_target, self.k_tau)
        soft_update(self.critic, self.critic_target, self.k_tau)

    def extract_state(self, observation):
        state = observation
        return state

    def network_eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def network_cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def random_action(self):
        action = np.random.uniform(
            self.min_a, self.max_a, self.num_actions)
        self.curr_a = action
        return action

    def select_action(self, state, trainable_flag=True, decay_epsilon=True):
        action = to_numpy(self.actor(to_var(np.array(state))))
        # Add noise in training stage
        action += trainable_flag * max(self.k_epsilon, 0) * self.random_process.sample()
        action = np.clip(action, self.min_a, self.max_a)
        if decay_epsilon:
            self.k_epsilon -= self.k_depsilon
        self.curr_a = action
        return action

    def reset(self, state):
        self.curr_s = state
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
