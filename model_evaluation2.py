import gymnasium as gym
from Agent import Agent

def make_env(envs_create):
    return gym.vector.SyncVectorEnv([lambda: gym.wrappers.FrameStack(
        gym.wrappers.AtariPreprocessing(gym.make("ALE/" + game + "-v5", frameskip=1)), 4) for _ in
                                     range(envs_create)])

if __name__ == "__main__":

    eval_envs = 4
    game = "BattleZone"

    print("About to create environments")

    eval_env = make_env(eval_envs)

    print("Finished creating eval envs")

