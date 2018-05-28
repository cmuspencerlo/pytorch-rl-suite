import torch
import torch.nn as nn
import torch.nn.functional as F
from util.distributions import get_distribution
from torch.nn.init import orthogonal

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

"""
All classes that inheret from Policy are expected to have
a feature extractor for actor and critic (see examples below)
and modules called linear_critic and dist. Where linear_critic
takes critic features and maps them to value and dist
represents a distribution of actions.
"""
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

    # def forward(self, inputs, states, masks):
    #     raise NotImplementedError

    def act(self, obs, states, masks, deterministic=False):
        hidden_critic, hidden_actor, states = self.forward(obs, states, masks)
        value = self.critic_linear(hidden_critic)
        action = self.action_distribution.sample(hidden_actor, deterministic=deterministic)
        action_log_probs, dist_entropy = self.action_distribution.logprobs_and_entropy(hidden_actor, action)
        return value, action, action_log_probs, states

    def compute_value(self, obs, states, masks):
        hidden_critic, _, states = self.forward(obs, states, masks)
        value = self.critic_linear(hidden_critic)
        return value

    def evaluate_actions(self, obs, states, masks, actions):
        hidden_critic, hidden_actor, states = self.forward(obs, states, masks)
        action_log_probs, dist_entropy = self.action_distribution.logprobs_and_entropy(hidden_actor, actions)
        value = self.critic_linear(hidden_critic)
        return value, action_log_probs, dist_entropy, states


class CNNPolicy(Policy):
    def __init__(self, in_channels, action_space, use_gru):
        super(CNNPolicy, self).__init__()
        # Declare as a 2D plane
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 32, 2, 1)
        self.linear1 = nn.Linear(32 * 8 * 8, 1024)
        if use_gru:
            self.gru = nn.GRUCell(1024, 1024)
        self.critic_linear = nn.Linear(1024, 1)
        # action_space = 6
        self.action_distribution = get_distribution(1024, action_space)

        self.train()
        self.reset_parameters()

    # Used as attribute outside
    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 1024
        else:
            return 1

    def reset_parameters(self):
        # Use init function
        self.apply(weights_init)

        # Standard init method
        # Only used in init
        # Based on activation used in next layer
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        if hasattr(self, 'gru'):
            orthogonal(self.gru.weight_ih.data)
            orthogonal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        if self.action_distribution.__class__.__name__ == "DiagGaussian":
            self.action_distribution.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        # Call it as a 4D object
        # What if I do not divide by 255?
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 8 * 8)
        x = self.linear1(x)
        x = F.relu(x)

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)
        return x, x, states


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MLPPolicy(Policy):
    def __init__(self, num_inputs, action_space):
        super(MLPPolicy, self).__init__()

        self.action_space = action_space

        self.a_fc1 = nn.Linear(num_inputs, 64)
        self.a_fc2 = nn.Linear(64, 64)

        self.v_fc1 = nn.Linear(num_inputs, 64)
        self.v_fc2 = nn.Linear(64, 64)

        self.critic_linear = nn.Linear(64, 1)
        self.dist = get_distribution(64, action_space)

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        """
        tanh_gain = nn.init.calculate_gain('tanh')
        self.a_fc1.weight.data.mul_(tanh_gain)
        self.a_fc2.weight.data.mul_(tanh_gain)
        self.v_fc1.weight.data.mul_(tanh_gain)
        self.v_fc2.weight.data.mul_(tanh_gain)
        """

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        x = self.v_fc1(inputs)
        x = F.tanh(x)

        x = self.v_fc2(x)
        hidden_critic = F.tanh(x)

        x = self.a_fc1(inputs)
        x = F.tanh(x)

        x = self.a_fc2(x)
        hidden_actor = F.tanh(x)

        return hidden_critic, hidden_actor, states
