import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from frozen_lake import FrozenLake


class GymTester(object):

    def __init__(self):
        pass

    def reward_visualizer(instances=20):
        def _reward_visualizer(gen_instance_rewards):
            def wrapper(self):
                plt.figure(0)
                all_rewards = []
                for i in range(instances):
                    all_rewards.append(gen_instance_rewards(self, i))
                fig = plt.imshow(all_rewards)
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                plt.show()
            return wrapper
        return _reward_visualizer

    @reward_visualizer(instances=20)
    def frozen_lake(self, i):
        qrl = FrozenLake(learning_rate=0.97 ** i, discount=0.99 ** i, iterations=1000)
        qrl.run()
        num_rewards = len(qrl.rewards)
        return self.rolling_max(qrl.rewards, window=num_rewards // 100, strides=num_rewards // 50)

    def rolling_mean(self, arr, window=2, strides=5):
        window = min(window, strides)
        return reduce(lambda acc, i: acc + [np.mean(arr[i-window:i+window])] if i % strides == 0 and not arr[i-window:i+window] == [] else acc, range(len(arr)), [])

    def rolling_max(self, arr, window=2, strides=5):
        window = min(window, strides)
        return reduce(lambda acc, i: acc + [np.max(arr[i-window:i+window])] if i % strides == 0 and not arr[i-window:i+window] == [] else acc, range(len(arr)), [])

    def inspect_rewards(self, rewards):
        num_rewards = len(rewards)
        print(self.rolling_max(rewards, window=num_rewards // 200, strides=num_rewards // 100))

    def visualize_rewards(self, rewards):
        plt.figure(0)
        num_rewards = len(rewards)
        fig = plt.imshow(np.atleast_2d(self.rolling_max(rewards, window=num_rewards // 200, strides=num_rewards // 100)))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()

gym_tester = GymTester()
gym_tester.frozen_lake()