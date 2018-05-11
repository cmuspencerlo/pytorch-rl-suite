import gym
from model.model import SimpleNet
import numpy as np
import torch.multiprocessing as mp
from util.util import *

class Worker(mp.Process):
    def __init__(self, args, global_net, global_opt, global_ep_step, ret_pool, id):
        super(Worker, self).__init__()

        self.id = 'w%i' % id
        self.max_w_episode = args.max_w_episode
        self.max_w_episode_step = args.max_w_episode_step
        self.global_net, self.global_opt = global_net, global_opt
        self.global_ep_step, self.ret_pool = global_ep_step, ret_pool
        self.env = gym.make(args.env_str).unwrapped
        N_S = self.env.observation_space.shape[0]
        N_A = self.env.action_space.shape[0]          # local network
        self.net = SimpleNet(N_S, N_A)             

    def run(self):
        total_step = 1
        for _ in range(self.max_w_episode):
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            epoch_reward = 0
            for step in range(self.max_w_episode_step):
                a = self.net.choose_action(to_var(s, volatile=True))
                s_, r, done, _ = self.env.step(a.clip(-2, 2))
                if step == 199:
                    done = True
                epoch_reward += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append((r + 8.1) / 8.1)    # normalize to [-1, 1]

                if total_step % 10 == 0 or done:
                    self.sync_global(done, s_, buffer_s, buffer_a, buffer_r)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    if done:
                        record(self.global_ep_step, self.ret_pool, epoch_reward, self.id)
                        break
                s = s_
                total_step += 1

    def sync_global(self, done, s_, buffer_s, buffer_a, buffer_r):

        # generate expected state_value td-error
        if done:
            v_td = 0
        else:
            v_td = self.net(to_var(s_, volatile=True))[-1].data.numpy()
            # v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

        buffer_v_td = []
        for r in buffer_r[::-1]:  # reverse order
            # generate td value
            v_td = r + 0.99 * v_td
            buffer_v_td.append(v_td)
        buffer_v_td.reverse()

        loss = self.net.loss_fn(
            to_var(np.vstack(buffer_s)), 
            to_var(np.vstack(buffer_a)), 
            to_var(np.vstack(buffer_v_td)))

        # How to avoid data racing in global_net         
        self.global_opt.zero_grad()
        self.net.zero_grad()
        loss.backward()
        for lp, gp in zip(self.net.parameters(), self.global_net.parameters()):
            gp._grad = lp.grad
        self.global_opt.step()

        # pull parameters
        self.net.load_state_dict(self.global_net.state_dict())
