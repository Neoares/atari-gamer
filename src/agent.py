import numpy as np
import random


class Agent(object):
    def __init__(self, state_size, action_size, prod=False):
        self.state_size = state_size
        self.action_size = action_size
        self.prod = prod

        self.alpha = .5
        self.alpha_min = 0.001
        # self.alpha_decay = 0.999999

        self.gamma = 0.99
        self.gamma_min = 0.1
        # self.gamma_decay = .999995

        self.epsilon = 1
        self.epsilon_min = 0.1
        # self.epsilon_decay = .999995

        self.Q_table = np.zeros((state_size, action_size))

    def update_q_table(self, state, action, reward, next_state, done):
        if not self.prod:
            self.Q_table[state, action] = self.Q_table[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q_table[next_state]) - self.Q_table[state, action])

    def act(self, state):
        if self.prod or random.random() > self.epsilon:
            return np.argmax(self.Q_table[state])
        else:
            return random.randrange(self.action_size)
