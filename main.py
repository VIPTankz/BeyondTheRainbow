
import numpy as np
import time
from copy import deepcopy
import sys
import torch
import gymnasium as gym
import wandb
from torch.profiler import profile, record_function, ProfilerActivity
def make_env(envs_create):
    return gym.vector.AsyncVectorEnv([lambda: gym.wrappers.FrameStack(
        gym.wrappers.AtariPreprocessing(gym.make("ALE/" + game + "-v5", frameskip=1)), 4) for _ in range(envs_create)],
                                     context="spawn")


if __name__ == '__main__':

    from Agent import Agent

    agent_name = "BTR_V100_32env_rr2_bs32_persplit8"

    # atari-3 : Battle Zone, Name This Game, Phoenix
    # atari-5 : Battle Zone, Double Dunk, Name This Game, Phoenix, Q*Bert

    testing = False

    if testing:
        num_envs = 4
    else:
        num_envs = 64

    n_steps = 50000000
    num_eval_episodes = 100
    eval_every = 1000000
    next_eval = eval_every

    gameset = ["BattleZone", "NameThisGame", "Phoenix", "DoubleDunk", "Qbert"]

    game_num = int(sys.argv[1])

    game = gameset[game_num]

    print("Currently Playing Game: " + str(game))

    gpu = sys.argv[2]
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    env = make_env(num_envs)

    print(env.observation_space)
    print(env.action_space[0])

    agent = Agent(n_actions=env.action_space[0].n, input_dims=[4, 84, 84], device=device, num_envs=num_envs,
                  agent_name=agent_name, total_frames=n_steps, testing=testing)


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
    last_steps = 0
    last_time = time.time()
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

        for stream in range(num_envs):
            if done_[stream]:
                observation[stream] = deepcopy(info["final_observation"][stream])

        if steps % 1200 == 0 and len(scores) > 0:

            avg_score = np.mean(scores_temp[-50:])

            if episodes % 1 == 0:
                print('{} {} avg score {:.2f} total_steps {:.0f} fps {:.2f}'
                      .format(agent_name, game, avg_score, steps, (steps - last_steps) / (time.time() - last_time)), flush=True)
                last_steps = steps
                last_time = time.time()

        # Evaluation
        if steps >= next_eval or steps >= n_steps:

            print("Evaluating")

            fname = agent_name + game + "Experiment.npy"
            np.save(fname, np.array(scores))

            eval_env = make_env(num_envs)

            agent.set_eval_mode()
            evals = []
            eval_episodes = 0
            eval_scores = np.array([0 for i in range(num_envs)])
            eval_observation, eval_info = eval_env.reset()

            while eval_episodes < num_eval_episodes:

                eval_action = agent.choose_action(eval_observation)  # this takes and return batches

                eval_observation_, eval_reward, eval_done_, eval_trun_, eval_info = eval_env.step(eval_action)

                # TRUNCATATION NOT IMPLEMENTED
                eval_done_ = np.logical_or(eval_done_, eval_trun_)

                for i in range(num_envs):
                    eval_scores[i] += eval_reward[i]

                    if eval_done_[i]:
                        eval_episodes += 1
                        wandb.log({"eval_scores": eval_scores[i]})
                        evals.append(eval_scores[i])
                        print("Eval Score: " + str(eval_scores[i]))

                        eval_scores[i] = 0

                eval_observation = deepcopy(eval_observation_)

                for stream in range(num_envs):
                    if eval_done_[stream]:
                        eval_observation[stream] = deepcopy(eval_info["final_observation"][stream])


            fname = agent_name + game + "Evaluation" + str(next_eval) + ".npy"
            np.save(fname, np.array(evals))
            next_eval += eval_every
            agent.set_train_mode()


wandb.finish()
