import numpy as np
import random
from collections import deque

from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam


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
        self.Q_table[state, action] = self.Q_table[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q_table[next_state]) - self.Q_table[state, action])

    def act(self, state):
        if self.prod or random.random() > self.epsilon:
            return np.argmax(self.Q_table[state])
        else:
            return random.randrange(self.action_size)


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
