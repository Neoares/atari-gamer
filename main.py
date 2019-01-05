import argparse

import gym
import numpy as np
import matplotlib.pyplot as plt

from src.agent import Agent, DQNAgent


def train(env, agent, episodes):
    total_epochs, total_penalties = 0, 0

    total_score = np.zeros(episodes)

    epsilon_decay = 2 * (agent.epsilon - agent.epsilon_min) / episodes
    alpha_decay = (agent.alpha - agent.alpha_min) / episodes
    gamma_decay = (agent.gamma - agent.gamma_min) / episodes

    for i in range(episodes):
        state = env.reset()
        epochs, penalties = 0, 0
        while True:
            epochs += 1
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            if reward == -10:
                penalties += 1
            agent.update_q_table(state, action, reward, next_state, done)
            # env.render()
            state = next_state
            total_score[i] += reward

            if done:
                if i%100 == 0:
                    print("episode {}/{}, score: {}, epsilon: {:.2}, gamma: {:.2}, alpha: {:.2}".format(i, episodes, total_score[i],
                            float(agent.epsilon), float(agent.gamma), float(agent.alpha)))
                break

        agent.epsilon = max(agent.epsilon_min, agent.epsilon - epsilon_decay)
        agent.alpha = max(agent.alpha_min, agent.alpha - alpha_decay)
        agent.gamma = max(agent.gamma_min, agent.gamma - gamma_decay)
        total_epochs += epochs
        total_penalties += penalties

    print("TOTAL SCORES:")
    print(total_score)
    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")
    plt.plot(range(episodes), total_score, 'bx')
    plt.show()

    # Close the env and write monitor result info to disk
    env.env.close()

    return total_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='Taxi-v2', help='Select the environment to run')
    args = parser.parse_args()

    env = gym.make(args.env_id)
    env.seed(42)
    agent = Agent(env.observation_space.n, env.action_space.n)

    train_score = train(env, agent, 100000)
    agent.prod = True
    test_score = train(env, agent, 1000)
    print("Production algorithm mean:", test_score.mean())
    print("Last 100 games mean:", test_score[-100:].mean())
