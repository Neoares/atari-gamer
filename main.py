import argparse

import gym
import numpy as np

from src.agent import Agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='KungFuMaster-v0', help='Select the environment to run')
    args = parser.parse_args()

    env = gym.make(args.env_id)
    env.seed(0)
    agent = Agent(env.action_space, random=False)

    episode_count = 10

    total_score = np.zeros(episode_count)

    for i in range(episode_count):
        ob = env.reset()
        agent.epsilon = max(1/(i+1), 0.1)
        while True:
            # action = agent.act(ob, reward, done)
            action = agent.act()
            ob, reward, done, _ = env.step(action)
            env.render()
            agent.update_values(action, reward)
            #   print(reward)
            if reward != 0:
                total_score[i] += reward
                #print("new action:", action)
                #print(agent.counts)
                #print(agent.values)
            if done:
                print(agent.counts)
                print(agent.values)
                print("final score:", total_score)
                break
            #input()
    print("TOTAL SCORES:")
    print(total_score)

    # Close the env and write monitor result info to disk
    env.env.close()
