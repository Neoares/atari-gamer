import argparse
import sys

import gym
from gym import wrappers, logger

import numpy as np
import random


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, random=False):
        self.action_space = action_space
        self.epsilon = 0.3
        self.counts = np.zeros(self.action_space.n)
        self.values = np.zeros(self.action_space.n)
        self.random = random

    def update_counts(self, action):
        self.counts[action] += 1

    def update_values(self, action, reward):
        value = self.values[action]
        n = self.counts[action]
        self.values[action] = value*(n-1)/n + reward/n

    def act(self):
        if not self.random and random.random() > self.epsilon:
            argmax = np.argwhere(self.values == np.amax(self.values)).ravel()
            new_action = argmax[random.randint(0, len(argmax)-1)]
        else:
            new_action = self.action_space.sample()
        self.update_counts(new_action)

        return new_action


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='KungFuMaster-v0', help='Select the environment to run')
    args = parser.parse_args()

    env = gym.make(args.env_id)
    env.seed(0)
    agent = RandomAgent(env.action_space, random=False)

    episode_count = 10
    reward = 0
    done = False

    total_score = np.zeros(episode_count)

    for i in range(episode_count):
        ob = env.reset()
        agent.epsilon = max(1/(i+1), 0.1)
        while True:
            action = agent.act()
            ob, reward, done, _ = env.step(action)
            env.render()
            agent.update_values(action, reward)
            total_score[i] += reward
            if done:
                print(agent.counts)
                print(agent.values)
                print("final score:", total_score)
                break
    print("TOTAL SCORES:")
    print(total_score)

    # Close the env and write monitor result info to disk
    env.env.close()
