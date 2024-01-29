
import numpy as np
import time
from copy import deepcopy
import sys
import torch
import gymnasium as gym
import wandb

def make_env(envs_create):
    return gym.vector.AsyncVectorEnv([lambda: gym.wrappers.FrameStack(
        gym.wrappers.AtariPreprocessing(gym.make("ALE/" + game + "-v5", frameskip=1)), 4) for _ in range(envs_create)])


if __name__ == '__main__':

    from Agent import Agent

    agent_name = "BTR_Trial"

    # atari-3 : Battle Zone, Name This Game, Phoenix
    # atari-5 : Battle Zone, Double Dunk, Name This Game, Phoenix, Q*Bert

    num_envs = 4
    n_steps = 100000 #50000000

    gameset = ["BattleZone", "NameThisGame", "Phoenix", "DoubleDunk", "Qbert"]

    game_num = int(sys.argv[1])

    game = gameset[game_num]

    num_eval_episodes = 5
    eval_every = 4000
    next_eval = eval_every

    print("Currently Playing Game: " + str(game))

    gpu = sys.argv[2]
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    env = make_env(num_envs)

    print(env.observation_space)
    print(env.action_space[0])

    agent = Agent(n_actions=env.action_space[0].n, input_dims=[4, 84, 84], device=device, num_envs=num_envs,
                  agent_name=agent_name, total_frames=n_steps)


    wandb.init(
        # set the wandb project where this run will be logged
        project="BeyondTheRainbow",

        # track hyperparameters and run metadata
        config={
            "agent_name": agent_name,
            "game": game,
            "steps": n_steps,
            "num_envs": num_envs,
            "batch_size": agent.batch_size,
            "IQN": agent.iqn,
            "munchausen": agent.munchausen,
            "impala": agent.impala,
            "model_size": agent.model_size,
            "noisy": agent.noisy,
            "per_alpha": agent.per_alpha,
            "discount": agent.gamma,
            "maxpool": agent.maxpool,
            "stabiliser": agent.stabiliser,
            "target_replace": agent.replace_target_cnt,
            "ema_tau": agent.soft_update_tau,
            "tr_alpha": agent.tr_alpha,
            "tr_period": agent.tr_period,
            "loss": agent.loss_type
        }
    )

    scores_temp = []
    steps = 0
    episodes = 0
    start = time.time()

    scores_count = [0 for i in range(num_envs)]
    scores = []
    done = False
    observation, info = env.reset()

    while steps < n_steps:
        steps += num_envs

        action = agent.choose_action(observation)  # this takes and return batches

        env.step_async(action)

        # this is placed here so learning takes place while step is happening
        agent.learn()

        observation_, reward, done_, trun_, info = env.step_wait()

        #TRUNCATATION NOT IMPLEMENTED
        done_ = np.logical_or(done_, trun_)

        for i in range(num_envs):
            scores_count[i] += reward[i]

            if done_[i]:
                episodes += 1
                scores.append([scores_count[i], steps])
                scores_temp.append(scores_count[i])
                wandb.log({"train_scores": scores_count[i], "steps": steps, "episodes": episodes,
                           "walltime": time.time() - start})

                scores_count[i] = 0

        reward = np.clip(reward, -1., 1.)

        for stream in range(num_envs):
            agent.store_transition(observation[stream], action[stream], reward[stream], done_[stream], stream=stream)

        observation = deepcopy(observation_)

        if steps % 1200 == 0 and len(scores) > 0:

            avg_score = np.mean(scores_temp[-50:])

            if episodes % 1 == 0:
                print('{} {} avg score {:.2f} total_steps {:.0f} fps {:.2f}'
                      .format(agent_name, game, avg_score, steps, steps / (time.time() - start)), flush=True)

        # Evaluation
        if steps >= next_eval or steps > n_steps:

            fname = agent_name + game + "Experiment.npy"
            np.save(fname, np.array(scores))

            eval_env = make_env(1)

            agent.set_eval_mode()
            evals = []
            eval_episodes = 0
            while eval_episodes < num_eval_episodes:
                eval_done = np.array([False])
                eval_observation, _ = eval_env.reset()
                eval_score = 0
                while not eval_done.any():
                    eval_action = agent.choose_action(observation)

                    eval_observation_, eval_reward, eval_done_, eval_trun_, eval_info = eval_env.step(eval_action)

                    # TRUNCATATION NOT IMPLEMENTED
                    eval_done = np.logical_or(eval_done_, eval_trun_)
                    eval_reward = eval_reward[0]

                    eval_score += eval_reward
                    eval_observation = eval_observation_

                evals.append(eval_score)
                wandb.log({"eval_scores": eval_score})
                print("Evaluation Score: " + str(eval_score))
                eval_episodes += 1
                print("Average:")
                print(np.mean(np.array(evals)))

            fname = agent_name + game + "Evaluation" + str(next_eval) + ".npy"
            np.save(fname, np.array(evals))
            next_eval += eval_every
            agent.set_train_mode()


    wandb.finish()
