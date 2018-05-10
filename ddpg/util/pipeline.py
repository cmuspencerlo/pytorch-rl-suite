#!/usr/bin/env python3

from copy import deepcopy
import numpy as np
from util.util import *

# from ipdb import set_trace as debug

class Pipeline(object):
    def __init__(self, args):
        self.global_step = 0
        self.episode = 0
        self.episode_step = 0 
        self.episode_reward = 0.

    def train(self, args, env, agent, debug=False):
        # writer = SummaryWriter()
        # agent.is_training = True
        observation = None
        state = None
        while self.episode < args.max_episode:
            # reset if it is the start of episode
            if observation is None:
                observation = deepcopy(env.reset())
                if state is None:
                    state = agent.extract_state(observation)
                agent.reset(state)

            # agent pick action ...
            if self.global_step <= args.warmup_step:
                action = agent.random_action()
            else:
                action = agent.select_action(state)

            observation, reward, done, info = env.step(action)
            state = agent.extract_state(observation)
            # observation2, reward, done, info = env.step(action)
            # observation2 = deepcopy(observation2)
            if self.episode_step >= args.max_episode_step:
                done = True

            # agent observe and update policy
            agent.observe(self.global_step, reward, state,
                done, self.global_step > args.warmup_step)


            # [optional] save intermideate model
            # if step % int(num_iterations/3) == 0:
            #     agent.save_model(output)

            # update
            self.global_step += 1
            self.episode_step += 1
            self.episode_reward += reward
            # observation = deepcopy(observation2)

            if done: # end of episode
                print('#{}: episode_reward:{} steps:{}'.format(
                    self.episode, self.episode_reward, self.episode_step))

                # Reactivate last state 
                agent.memory.append(state, agent.select_action(observation), 0., False)

                # evaluate step
                if self.episode % args.validate_trigger_episode == 0:
                    validate_reward = self.evaluate(args, env, agent, debug=False, visualize=False, save=True)
                    if debug:
                        print_yellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(self.global_step, validate_reward))

                # reset
                observation = None
                state = None
                self.episode += 1
                self.episode_step = 0
                self.episode_reward = 0.
        
    def test(self, num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):
        agent.load_weights(model_path)
        agent.is_training = False
        agent.eval()
        policy = lambda x: agent.select_action(x, decay_epsilon=False)

        for i in range(num_episodes):
            validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
            if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))

    def evaluate(self, args, env, agent, debug=False, visualize=False, save=True):
        observation = None
        result = []
        for episode in range(args.validate_episode):
            observation = env.reset()
            state = agent.extract_state(observation)
            episode_step = 0
            episode_reward = 0.
            assert state is not None

            # start episode
            done = False
            while not done:
                action = agent.select_action(state, False, False)
                observation, reward, done, info = env.step(action)
                state = agent.extract_state(observation)
                if episode_step >= args.max_episode_step:
                    done = True
                # if visualize:
                #     env.render(mode='human')

                # update
                episode_reward += reward
                episode_step += 1

            # if args.debug: 
            #     print_yellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            result.append(episode_reward)
        return np.mean(np.array(result))

        # result = np.array(result).reshape(-1,1)
        # self.results = np.hstack([self.results, result])
