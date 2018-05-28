import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

NUM_PROCESSES = torch.multiprocessing.cpu_count()

class RolloutStorage(object):
    def __init__(self, num_steps, obs_shape, action_space, state_size):
        self.observations = torch.zeros(num_steps + 1, NUM_PROCESSES, *obs_shape)
        self.states = torch.zeros(num_steps + 1, NUM_PROCESSES, state_size)
        # why
        self.rewards = torch.zeros(num_steps, NUM_PROCESSES, 1)
        self.action_log_probs = torch.zeros(num_steps, NUM_PROCESSES, 1)

        # This should be a q-value prediction
        self.value_preds = torch.zeros(num_steps + 1, NUM_PROCESSES, 1)
        self.values = torch.zeros(num_steps + 1, NUM_PROCESSES, 1)
        self.masks = torch.ones(num_steps + 1, NUM_PROCESSES, 1)

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, NUM_PROCESSES, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.num_steps = num_steps
        self.index = 0

    def to_cuda(self):
        self.observations = self.observations.cuda()
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.values = self.values.cuda()
        self.value_preds = self.value_preds.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, current_obs, state, mask, action, action_log_prob, reward, value_pred):
        self.observations[self.index + 1].copy_(current_obs)
        self.states[self.index + 1].copy_(state)
        self.masks[self.index + 1].copy_(mask)

        self.actions[self.index].copy_(action)
        self.action_log_probs[self.index].copy_(action_log_prob)
        self.rewards[self.index].copy_(reward)
        self.value_preds[self.index].copy_(value_pred)

        self.index = (self.index + 1) % self.num_steps

    def post_processing(self):
        # Closed-loop setting
        self.observations[0].copy_(self.observations[-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_value_preds(self, next_state_value, use_gae, GAMMA, TAU):
        if use_gae:
            self.values[-1] = next_state_value
            gae = 0
            for index in reversed(range(self.rewards.size(0))):
                delta = self.rewards[index] + GAMMA * self.masks[index + 1] * self.values[index + 1] - self.values[index]
                gae = delta + GAMMA * TAU * self.masks[index + 1] * gae
                self.value_preds[index] = gae + self.values[index]
        else:
            self.value_preds[-1] = next_state_value
            for index in reversed(range(self.rewards.size(0))):
                self.value_preds[index] = self.rewards[index] + \
                    GAMMA * self.masks[index + 1] * self.value_preds[index + 1]

    def feed_forward_generator(self, advantages, num_batch):
        num_steps = self.rewards.size()[0]
        data_size = num_steps * NUM_PROCESSES
        batch_size = data_size // num_batch
        batch_sampler = BatchSampler(SubsetRandomSampler(range(data_size)), batch_size, drop_last=False)
        for indices in batch_sampler:
            # indices = [89, 7, 35]
            # index => list
            # indices = torch.cuda.LongTensor(indices)

            # Truncate part
            observations_batch = self.observations[:-1].view(-1, \
                                        *self.observations.size()[2:])[indices]
            states_batch = self.states[:-1].view(-1, self.states.size(-1))[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]

            # Non-truncate part
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_target = advantages.view(-1, 1)[indices]

            yield observations_batch, states_batch, actions_batch, \
                value_preds_batch, masks_batch, old_action_log_probs_batch, adv_target


    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = []
            states_batch = []
            actions_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                observations_batch.append(self.observations[:-1, ind])
                states_batch.append(self.states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            observations_batch = torch.cat(observations_batch, 0)
            states_batch = torch.cat(states_batch, 0)
            actions_batch = torch.cat(actions_batch, 0)
            return_batch = torch.cat(return_batch, 0)
            masks_batch = torch.cat(masks_batch, 0)
            old_action_log_probs_batch = torch.cat(old_action_log_probs_batch, 0)
            adv_targ = torch.cat(adv_targ, 0)

            yield observations_batch, states_batch, actions_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, adv_targ
