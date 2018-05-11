import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

class SimpleNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(SimpleNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 100)
        self.mu = nn.Linear(100, 1)
        self.sigma = nn.Linear(100, 1)
        self.c1 = nn.Linear(s_dim, 100)
        self.v = nn.Linear(100, 1)
        layers = [self.a1, self.mu, self.sigma, self.c1, self.v]
        for layer in layers:                                                                                                                                                                                
            nn.init.normal(layer.weight, mean=0., std=0.1)                                                                                                                                                  
            nn.init.constant(layer.bias, 0.1) 

    def forward(self, s):
        a1 = F.relu(self.a1(s))
        # mu = 2 * F.tanh(self.mu(a1))
        mu = F.tanh(self.mu(a1))
        # sigma = F.softplus(self.sigma(a1)) + 0.001      # avoid 0
        sigma = F.relu(self.sigma(a1)) + 0.001
        c1 = F.relu(self.c1(s))
        value = self.v(c1)
        return mu, sigma, value

    def choose_action(self, s):
        mu, sigma, _ = self.forward(s)
        # m = self.distribution(mean=mu.view(1, ).data, std=sigma.view(1, ).data)
        # gauss = D.Normal(mu.data, sigma.data)
        gauss = D.Normal(mu, sigma)
        return gauss.sample().data.numpy()

    def loss_fn(self, s, a, v_td):
        # critic loss
        mu, sigma, value = self.forward(s)
        td_error = v_td - value
        critic_loss = td_error.pow(2)

        # policy gradient loss for actor
        gauss = D.Normal(mu, sigma)
        log_prob = gauss.log_prob(a)
        entropy = torch.log(gauss.std)  # exploration
        # classic actor gradient * reward
        actor_loss = -(log_prob * td_error.detach() + 0.001 * entropy)

        # combine together
        return (critic_loss + actor_loss).mean()