import torch

import stratego_gym
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import time

from deep_nash.agent import DeepNashAgent
from deep_nash.network import DeepNashNet


def basic_test():
    env = gym.make("stratego_gym/Stratego-v0", render_mode="human")
    env.reset()

    start_time = time.time()
    count = 0
    while time.time() - start_time < 60:
        action = env.unwrapped.get_random_action()
        state, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.1)
        count += 1
        if terminated:
            print(f"Game over! Player {-1 * info['cur_player']} received {reward}")
            env.reset()
    print(count)

def policy_test():
    device = "cuda"
    agent = DeepNashAgent(device)

    env = gym.make("stratego_gym/Stratego-v0", render_mode="human")
    state, info = env.reset()

    start_time = time.time()
    count = 0
    while time.time() - start_time < 60:
        obs, mask = state['obs'], state['action_mask']
        action_coord = agent.get_action(obs, mask, info)
        state, reward, terminated, truncated, info = env.step(action_coord)
        env.render()
        time.sleep(0.1)
        count += 1
        if terminated:
            print(f"Game over! Player {-1 * info['cur_player']} received {reward}")
            state, info = env.reset()
    print(count)


def vectorized_test(num_envs):
    device = "cuda"
    policy = DeepNashNet(10, 20, 0, 0).to(device)

    envs = AsyncVectorEnv([lambda env_id=env_id: gym.make("stratego_gym/Stratego-v0",
                                            render_mode="human" if env_id == 0 else None) for env_id in range(num_envs)],
                                            copy=False)
    state, infos = envs.reset()
    obs, masks = state['obs'], state['action_mask']

    start_time = time.time()
    count = 0

    while time.time() - start_time < 60:
        with torch.no_grad():
            # Convert observations to tensor
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

            # Forward pass
            (deploy_policy, _), (select_policy, _), (move_policy, _) = policy.forward(obs_t, infos)

            # Extract phases and shape info
            game_phases = infos["game_phase"]
            H, W = infos["board_shape"][0]


            # Boolean indexing for phases
            phase0_idx = (game_phases == 0)
            phase1_idx = (game_phases == 1)
            phase2_idx = (game_phases == 2)

            # Convert policies to numpy and select the appropriate policy per environment
            deploy_np = deploy_policy.squeeze(1).cpu().numpy()
            select_np = select_policy.squeeze(1).cpu().numpy()
            move_np = move_policy.squeeze(1).cpu().numpy()

            actions = (phase0_idx[:, None, None] * deploy_np +
                       phase1_idx[:, None, None] * select_np +
                       phase2_idx[:, None, None] * move_np)

            # Apply masks and normalize
            actions *= masks
            sums = actions.sum(axis=(1, 2), keepdims=True)
            sums[sums < 1e-8] = 1e-8
            actions /= sums

            # Sample actions
            flat_actions = actions.reshape(num_envs, -1)
            rand_vals = np.random.rand(num_envs)
            cumulative_sums = np.cumsum(flat_actions, axis=1)
            sampled_indices = np.argmax(cumulative_sums >= rand_vals[:, None], axis=1)

            rows, cols = divmod(sampled_indices, W)
            action_coords = np.stack([rows, cols], axis=1)

        state, rewards, terminated, truncated, infos = envs.step(action_coords)
        obs, masks = state['obs'], state['action_mask']
        envs.call("render")
        count += len(rewards)

    print(count)
    envs.close()


if __name__ == '__main__':
    # for i in range(2, 3):
    #     print(f"Vectorized test with {i} environments")
    #     vectorized_test(i)
    policy_test()
