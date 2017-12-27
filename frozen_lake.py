import numpy as np
import gym
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
import time

class FrozenLake(object):

    def __init__(self, learning_rate=0.8, discount=0.95, iterations=1000):
        # game environment
        self.env = gym.make('FrozenLake-v0')

        # Q table placeholder initialized with zeros
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])

        # learning hyperparameters
        self.discount = discount
        self.learning_rate = learning_rate

        # iterations / playthroughs / episodes / env runs
        self.iterations = iterations

        # track rewards
        self.rewards = []
        self.cur_reward = None

        # placeholder to store current environment state
        self.state = None

    def timeit(func):
        def wrapper(self):
            t = time.time()
            func(self)
            print(time.time() - t)
        return wrapper

    @timeit
    def run(self):
        for i in range(self.iterations):
            self.state = self.env.reset()
            acc_reward = 0
            goal_met_or_lost = False

            while not goal_met_or_lost:
                action = self.choose_best_next_action(i)
                state, reward, goal_met_or_lost, _ = self.env.step(action)
                self.update_q_table(state=state, action=action, reward=reward)
                self.state = state
                acc_reward += int(reward)
            self.rewards.append(acc_reward)

    def update_q_table(self, state=None, action=None, reward=None):
        self.q_table[self.state, action] = self.q_table[self.state, action] + self.learning_rate * self.future_reward_given_current(state, action, reward)

    def choose_best_next_action(self, iteration):
        return np.argmax(self.q_table[self.state, :] + np.random.randn(1, self.env.action_space.n) * (1. / (iteration + 1)))

    def future_reward_given_current(self, state, action, reward):
        return (reward + self.discount * np.max(self.q_table[state, :])) - self.q_table[self.state, action]





