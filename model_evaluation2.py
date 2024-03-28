import gymnasium as gym
from Agent import Agent
import argparse
import torch


def make_env(envs_create):
    return gym.vector.SyncVectorEnv([lambda: gym.wrappers.FrameStack(
        gym.wrappers.AtariPreprocessing(gym.make("ALE/" + game + "-v5", frameskip=1)), 4) for _ in
                                     range(envs_create)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--game', type=str, default="BattleZone")
    parser.add_argument('--agent_name', type=str, default="FinalAgent")

    eval_envs = 4
    args = parser.parse_args()

    game = args.game
    name_ending = args.agent_name
    frame_name = "50M"
    agent_name = "BTR_" + game + frame_name + "_" + name_ending

    print("Agent Name:" + str(agent_name))

    print("About to create environments")

    eval_env = make_env(eval_envs)

    print("Finished creating eval envs")

    print(eval_env.observation_space)
    print(eval_env.action_space[0])

    gpu = "0"
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    agent = Agent(n_actions=eval_env.action_space[0].n, input_dims=[4, 84, 84], device=device,
                  num_envs=eval_envs, agent_name=agent_name, total_frames=1000000)

    print("Created Agent")