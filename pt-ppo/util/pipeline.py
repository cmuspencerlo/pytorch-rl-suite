#!/usr/bin/env python3

from copy import deepcopy
import numpy as np
from model.model import *
import torch
import torch.optim as optim
from util.storage import RolloutStorage
from util.util import *

NUM_PROCESSES = torch.multiprocessing.cpu_count()
EPS = 1e-6

class Pipeline(object):
    def __init__(self, args):
        self.args = args

    # def train(self, env, agent, debug=False):
    def train(self, envs):
        obs_shape = envs.observation_space.shape
        # With tuple (4, 84, 84)
        # Without* (4, (84, 84))
        stack_obs_shape = (obs_shape[0] * self.args.num_stack, *obs_shape[1:])

        # action_space = Discrete(6) 0 ~ 5

        if len(envs.observation_space.shape) == 3:
            actor_critic = CNNPolicy(stack_obs_shape[0], envs.action_space, self.args.recurrent_policy).cuda()
        optimizer = optim.Adam(actor_critic.parameters(), self.args.lr, eps=self.args.eps)
        # else:
        #     assert not args.recurrent_policy, \
        #         "Recurrent policy is not implemented for the MLP controller"
        #     actor_critic = MLPPolicy(obs_shape[0], envs.action_space)
        # print(envs.action_space.__class__.__name__)
        if envs.action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = envs.action_space.shape[0]

        rollout_buffers = RolloutStorage(self.args.num_steps, stack_obs_shape, envs.action_space, actor_critic.state_size)
        current_obs = torch.zeros(torch.multiprocessing.cpu_count(), *stack_obs_shape)

        # why
        def update_current_obs(obs):
            shape_dim0 = envs.observation_space.shape[0]
            obs = torch.from_numpy(obs).float()
            # print(current_obs[:, :-1])
            if self.args.num_stack > 1:
                current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
            current_obs[:, -shape_dim0:] = obs

        # obs: (num_proc, 1, 84, 84)
        obs = envs.reset()
        update_current_obs(obs)
        rollout_buffers.observations[0].copy_(current_obs)

        # These variables are used to compute average rewards for all processes.
        episode_rewards = torch.zeros([NUM_PROCESSES, 1])
        final_rewards = torch.zeros([NUM_PROCESSES, 1])

        #     current_obs = current_obs.cuda()
        #     rollouts.cuda()
        num_updates = int(self.args.num_frames) // self.args.num_steps // torch.multiprocessing.cpu_count()
        for i in range(num_updates):
            for step in range(self.args.num_steps):
            # Sample actions
                value, action, action_log_prob, states = actor_critic.act(
                    to_var(rollout_buffers.observations[step], volatile=True),
                    to_var(rollout_buffers.states[step], volatile=True),
                    to_var(rollout_buffers.masks[step], volatile=True))

                actions = action.data.squeeze(1).cpu().numpy()
                obs, reward, done, _ = envs.step(actions)
                reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
                # reward (20, 1)
                episode_rewards += reward

                # If done then clean the history of observations.
                # why
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                print('kaka {}:{}'.format(i, final_rewards))
                episode_rewards *= masks

                if current_obs.dim() == 4:
                    current_obs *= masks.unsqueeze(2).unsqueeze(2)
                else:
                    current_obs *= masks

                update_current_obs(obs)
                rollout_buffers.insert(current_obs, states.data, masks, action.data, action_log_prob.data, reward, value.data)

            next_state_value = actor_critic.compute_value(
                                    to_var(rollout_buffers.observations[-1], volatile=True),
                                    to_var(rollout_buffers.states[-1], volatile=True),
                                    to_var(rollout_buffers.masks[-1], volatile=True)).data

            rollout_buffers.compute_value_preds(next_state_value, self.args.use_gae, self.args.gamma, self.args.tau)

            advantages = rollout_buffers.value_preds[:-1] - rollout_buffers.values[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

            for _ in range(self.args.ppo_epoch):
                if self.args.recurrent_policy:
                    data_generator = rollout_buffers.recurrent_generator(advantages, self.args.num_batch)
                else:
                    data_generator = rollout_buffers.feed_forward_generator(advantages, self.args.num_batch)

                for sample in data_generator:
                    observations_batch, states_batch, actions_batch, \
                        value_preds_batch, masks_batch, old_action_log_probs_batch, adv_target = sample

                    # why?
                    # Need a beautiful way
                    values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(
                            to_var(observations_batch), to_var(states_batch),
                            to_var(masks_batch), to_var(actions_batch).type(torch.cuda.LongTensor))

                    adv_target = to_var(adv_target)
                    ratio = torch.exp(action_log_probs - to_var(old_action_log_probs_batch))
                    surr1 = ratio * adv_target
                    surr2 = torch.clamp(ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * adv_target
                    action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)
                    critic_loss = (to_var(value_preds_batch) - values).pow(2).mean()

                    optimizer.zero_grad()
                    (critic_loss + action_loss - dist_entropy * self.args.entropy_coef).backward()
                    nn.utils.clip_grad_norm(actor_critic.parameters(), self.args.max_grad_norm)
                    optimizer.step()

            rollout_buffers.post_processing()

    #     if j % args.save_interval == 0 and args.save_dir != "":
    #         save_path = os.path.join(args.save_dir, args.algo)
    #         try:
    #             os.makedirs(save_path)
    #         except OSError:
    #             pass

    #         # A really ugly way to save a model to CPU
    #         save_model = actor_critic
    #         if args.cuda:
    #             save_model = copy.deepcopy(actor_critic).cpu()

    #         save_model = [save_model,
    #                         hasattr(envs, 'ob_rms') and envs.ob_rms or None]

    #         torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

            if i % self.args.log_interval == 0:
                total_num_steps = (i + 1) * NUM_PROCESSES * self.args.num_steps
                print("Updates {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                    format(i,
                           final_rewards.mean(),
                           final_rewards.median(),
                           final_rewards.min(),
                           final_rewards.max(), dist_entropy.data[0],
                           critic_loss.data[0], action_loss.data[0]))
                # writer.add_scalar('data/reward', final_rewards.mean(), j)
    #     if args.vis and j % args.vis_interval == 0:
    #         try:
    #             # Sometimes monitor doesn't properly flush the outputs
    #             win = visdom_plot(viz, win, args.log_dir, args.env_name,
    #                               args.algo, args.num_frames)
    #         except IOError:
    #             pass

    # writer.close()