import numpy as np
import random
from collections import deque

from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam


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


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1  # exploration rate (set to 1)
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(Conv2D(32, (7, 7), padding='same', input_shape=self.state_size, activation='relu'))
        model.add(Conv2D(32, (7, 7), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (5, 5), padding='same', input_shape=self.state_size, activation='relu'))
        model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding='same', input_shape=self.state_size, activation='relu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', input_shape=self.state_size, activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))

        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def act(self, state):
        # select random action with prob=epsilon else action=maxQ
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        # sample random transitions
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                Q_next = self.model.predict(next_state)[0]
                target = (reward + self.gamma * np.amax(Q_next))

            target_f = self.model.predict(state)
            target_f[0][action] = target

            # train network
            self.model.fit(state, target_f, epochs=1, verbose=0)
