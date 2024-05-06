import stratego_gym
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import time


def basic_test():
    env = gym.make("stratego_gym/Stratego-v0")
    env.reset()

    start_time = time.time()
    count = 0
    while time.time() - start_time < 60:
        action = env.unwrapped.get_random_action()
        state, reward, terminated, truncated, info = env.step(action)
        # env.render()
        count += 1
        if terminated:
            # print("Game Over", "Reward: ", reward)
            env.reset()
    print(count)


def vectorized_test(i):
    envs = AsyncVectorEnv([lambda: gym.make("stratego_gym/Stratego-v0") for _ in range(i)], copy=False)
    obs, infos = envs.reset()

    start_time = time.time()
    count = 0
    while time.time() - start_time < 60:
        actions = np.array(envs.call("get_random_action"))
        states, rewards, terminated, truncated, infos = envs.step(actions)
        count += len(rewards)

    print(count)


if __name__ == '__main__':
    for i in range(1, 14):
        print(f"Vectorized test with {i} environments")
        vectorized_test(i)
