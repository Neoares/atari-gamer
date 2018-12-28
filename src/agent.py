import numpy as np
import random


class Agent(object):
    def __init__(self, action_space, random=False):
        self.action_space = action_space
        self.epsilon = 0.3
        self.counts = np.zeros(self.action_space.n)
        self.values = np.zeros(self.action_space.n)
        #self.Q = np.zeros()
        self.random = random

    def update_counts(self, action):
        self.counts[action] += 1

    def update_values(self, action, reward):
        value = self.values[action]
        n = self.counts[action]
        #print("n:", n)
        #print(value*(n-1))
        #print(reward/n)
        self.values[action] = value*(n-1)/n + reward/n

    def act(self):
        if not self.random and random.random() > self.epsilon:
            argmax = np.argwhere(self.values == np.amax(self.values)).ravel()
            new_action = argmax[random.randint(0, len(argmax)-1)]
        else:
            new_action = self.action_space.sample()
        #print("new action:", new_action)
        self.update_counts(new_action)

        return new_action
