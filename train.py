import time
from random import random

import numpy as np
import torch
from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.collectors import MultiaSyncDataCollector, SyncDataCollector, MultiSyncDataCollector
from torchrl.collectors.distributed import RayCollector
from torchrl.envs import default_info_dict_reader, ParallelEnv
from torchrl.envs.libs.gym import GymEnv

from deep_nash.network import PyramidModule, ConvResBlock, DeconvResBlock
from deep_nash.rnad import RNaDSolver, RNaDConfig
from stratego_gym.envs.stratego import GAME_PHASE_DICT, DEPLOYMENT_PHASE, MOVEMENT_PHASE

# If deep_nash is an installable package, you can do:
# ray_init_config = {"runtime_env": {"pip": ["deep_nash"]}}

# If deep_nash is a local folder/package not on PyPI, provide its path:
# e.g., "py_modules": ["absolute/path/to/deep_nash_folder"]
ray_init_config = {
    "runtime_env": {
        # pick the approach that matches your setup:
        # "pip": ["deep_nash"],  # if it's installable from PyPI or local file
        "py_modules": ["/home/abhinav-peri/PycharmProjects/DeepNash/deep_nash",
                       "/home/abhinav-peri/PycharmProjects/DeepNash/stratego_gym"],  # if local
    },
}

from deep_nash import DeepNashAgent  # must be importable on the driver side too

# 1. Create environment factory
def env_maker(render_mode=None):
    reader = default_info_dict_reader(["cur_player", "game_phase"])
    return GymEnv("stratego_gym/Stratego-v0", render_mode=None).set_info_dict_reader(reader)

@torch.inference_mode
def evaluate_random(policy: TensorDictModule):
    env = env_maker().to("cuda")
    env.set_info_dict_reader(default_info_dict_reader(["cur_player", "game_phase", "cur_board"]))
    win_count = 0
    draw_count = 0
    for i in range(100):
        tensordict = env.reset()
        policy_turn = np.random.choice([1, -1])
        print(f"New Game! {i}")
        while True:
            if i < 10: print("------------------------")
            if i < 10: print(f"Game Phase: {GAME_PHASE_DICT[tensordict['game_phase'].item()]}")
            if i < 10: print(f"Cur Player: {tensordict['cur_player'].item()}")
            game_phase = tensordict['game_phase'].item()

            if tensordict["cur_player"] == policy_turn:
                tensordict = policy(tensordict)
                if i < 10: print(f"Value: {tensordict['value'].item()}")
                if i < 10: print(tensordict["policy"])
                tensordict = env.step(tensordict)["next"]
            else:
                if i < 10: print("Random Action")
                tensordict["action"] = env.action_spec.sample()
                tensordict = env.step(tensordict)["next"]

            if game_phase in (DEPLOYMENT_PHASE, MOVEMENT_PHASE):
                if i < 10: print(tensordict["cur_board"])

            if tensordict["terminated"]:
                if i < 10: print(f"Won Game? {-policy_turn == tensordict['cur_player']}")
                if i < 10: print(f"Draw Game? {(tensordict['reward'] == 0).item()}")
                win_count += int(-policy_turn == tensordict["cur_player"]) * tensordict["reward"].item()
                draw_count += 1 - tensordict["reward"].item()
                break

    env.close()

    print(f"Win Rate against Random: {win_count / 100}")
    print(f"Draw Rate against Random: {draw_count / 100}")
    print(f"Loss Rate against Random: {(100 - (win_count + draw_count)) / 100}")

    # env = env_maker(render_mode="human").to("cuda")
    # tensordict = env.reset()
    #
    # # Randomly select whether random or policy goes first
    # policy_turn = np.random.choice([True, False])
    #
    # while True:
    #     if policy_turn:
    #         tensordict = policy(tensordict)
    #         tensordict = env.step(tensordict)["next"]
    #     else:
    #         tensordict["action"] = env.action_spec.sample()
    #         tensordict = env.step(tensordict)["next"]
    #     env.render()
    #     time.sleep(0.1)
    #     if tensordict["terminated"]:
    #         print(f"Game over! " + ("Policy" if policy_turn else "Random") + " won.")
    #         tensordict = env.reset()

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    # 2. Define a policy
    # policy = DeepNashAgent()
    policy = torch.load("DeepNashPolicy.pt").cpu()
    env = env_maker()
    policy(env.reset())
    policy.to("cuda")

    # # Choose how many remote collectors and vectorized envs you want
    # num_collectors = 1
    # num_workers_per_collector = 10
    #
    # # frames_per_batch + total_frames
    # frames_per_batch = 10_000
    # total_frames = 10_000_000
    #
    # # 3. Remote resource config for each actor
    # remote_configs = {
    #     "num_cpus": 1,
    #     "num_gpus": 0.01,
    #     "memory": 2 * 1024 ** 3,  # 2GB
    # }
    #
    # torchrl_logger.info("Initializing the collector with ray_init_config")
    #
    # distributed_collector = RayCollector(
    #     create_env_fn=[env_maker] * num_collectors,
    #     policy=policy,
    #     frames_per_batch=frames_per_batch,
    #     total_frames=total_frames,
    #     sync=False,  # Asynchronous
    #     num_collectors=num_collectors,
    #     num_workers_per_collector=num_workers_per_collector,
    #     remote_configs=remote_configs,
    #     device=torch.device("cuda"),  # or "cpu"
    #     ray_init_config=ray_init_config,  # <-- The key line
    # )
    #
    # total_collected = 0
    # for batch_idx, batch_data in enumerate(distributed_collector, start=1):
    #     batch_size = batch_data.numel()
    #     total_collected += batch_size
    #     torchrl_logger.info(
    #         f"[Batch {batch_idx}] Received {batch_size:,} frames | "
    #         f"Total so far: {total_collected:,}/{total_frames:,}"
    #     )
    #     if total_collected >= total_frames:
    #         break
    #
    # torchrl_logger.info("Data collection complete!")

    # Define the number of collectors and workers per collector
    num_collectors = 6
    workers_per_collector = 6
    envs = [ParallelEnv(workers_per_collector, env_maker) for _ in range(num_collectors)]

    # Initialize the MultiaSyncDataCollector
    collector = MultiaSyncDataCollector (
        create_env_fn=envs,
        policy=policy,
        frames_per_batch=5120,
        total_frames=5120 * 100000,
        device="cuda",
        update_at_each_batch=True,
        split_trajs=True,
    )

    solver = RNaDSolver(policy, RNaDConfig(game_name="stratego"))
    for i, data in enumerate(collector):
        # Process the collected data
        # breakpoint()
        print("###############################################################")
        print("###############################################################")
        print("###############################################################")
        print(f"Batch {i} collected with shape: {data.batch_size}")
        logs = solver.step(data)
        print("Loss: " + str(logs["total loss"]) + " Value Loss: " + str(logs["loss_v"]) +
              " Policy Loss: " + str(logs["loss_nerd"]))

        if (i+1) % 100 == 0:
            print("################### Evaluating ###################")
            evaluate_random(policy)
            torch.save(policy, "DeepNashPolicy.pt")


    collector.shutdown()
    for name, param in policy.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.data}")
    breakpoint()