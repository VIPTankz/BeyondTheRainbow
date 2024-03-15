
import numpy as np
import time
from copy import deepcopy
import sys
import torch
import gymnasium as gym
import wandb
import os
import argparse, sys
from Agent import Agent
from torch.profiler import profile, record_function, ProfilerActivity
def make_env(envs_create):
    return gym.vector.AsyncVectorEnv([lambda: gym.wrappers.FrameStack(
        gym.wrappers.AtariPreprocessing(gym.make("ALE/" + game + "-v5", frameskip=1)), 4) for _ in range(envs_create)],
                                     context="spawn")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default="BattleZone")  # This is BattleZone
    parser.add_argument('--envs', type=int, default=64)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--rr', type=int, default=1)
    parser.add_argument('--frames', type=int, default=40000000)

    parser.add_argument('--maxpool_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--ema_tau', type=float, default=2.5e-4)
    parser.add_argument('--tr', type=int, default=0)

    # the way parser.add_argument handles bools in dumb so we use int 0 or 1 instead
    parser.add_argument('--noisy', type=int, default=0)
    parser.add_argument('--spectral', type=int, default=1)
    parser.add_argument('--iqn', type=int, default=1)

    parser.add_argument('--impala', type=int, default=1)
    parser.add_argument('--discount', type=float, default=0.997)
    parser.add_argument('--adamw', type=int, default=0)
    parser.add_argument('--lr_decay', type=int, default=0)
    parser.add_argument('--per', type=int, default=1)
    parser.add_argument('--taus', type=int, default=8)

    # features still in testing
    parser.add_argument('--pruning', type=int, default=0) # ONLY WORKS FOR DUELING
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--linear_size', type=int, default=512)
    parser.add_argument('--munch', type=int, default=1)
    parser.add_argument('--ema', type=int, default=0)
    parser.add_argument('--c', type=int, default=500)  # this is the target replace
    parser.add_argument('--model_size', type=int, default=2)

    # depends on munchausen
    parser.add_argument('--double', type=int, default=0)

    # likely dead improvements
    parser.add_argument('--sqrt', type=int, default=0)
    parser.add_argument('--ede', type=int, default=0)
    parser.add_argument('--discount_anneal', type=int, default=0)
    parser.add_argument('--moe', type=int, default=0)  # This Does not Work Yet!

    args = parser.parse_args()

    game = args.game
    envs = args.envs
    bs = args.bs
    rr = args.rr
    ema = args.ema
    tr = args.tr
    c = args.c
    ema_tau = args.ema_tau
    lr = args.lr

    maxpool_size = args.maxpool_size

    noisy = args.noisy
    spectral = args.spectral
    munch = args.munch
    iqn = args.iqn
    double = args.double

    dueling = args.dueling
    impala = args.impala
    discount = args.discount

    linear_size = args.linear_size

    adamw = args.adamw
    sqrt = args.sqrt
    ede = args.ede
    moe = args.moe
    lr_decay = args.lr_decay

    per = args.per
    taus = args.taus
    pruning = args.pruning
    model_size = args.model_size

    frames = args.frames

    discount_anneal = args.discount_anneal

    # atari-3 : Battle Zone, Name This Game, Phoenix
    # atari-5 : Battle Zone, Double Dunk, Name This Game, Phoenix, Qbert

    lr_str = "{:e}".format(lr)
    lr_str = str(lr_str).replace(".", "").replace("0", "")
    frame_name = str(int(frames / 1000000)) + "M"

    agent_name = "BTR_" + game + frame_name + "_lr" + lr_str + "_WD" + str(adamw) + "_LRD" + str(lr_decay) +\
                 "_lin_size" + str(linear_size) + "_dueling" + str(dueling)

    print("Agent Name:" + str(agent_name))
    testing = args.testing
    wandb_logs = not testing

    if wandb_logs:
        ###################### Making Dir Code
        # Initialize a counter to keep track of the suffix
        counter = 0

        # Loop until you find a directory name that doesn't exist
        while True:
            # Construct the directory name with the current counter
            if counter == 0:
                new_dir_name = agent_name
            else:
                new_dir_name = f"{agent_name}_{counter}"

            # Check if the directory already exists
            if not os.path.exists(new_dir_name):
                break

            # If it exists, increment the counter and try again
            counter += 1

        os.mkdir(new_dir_name)
        print(f"Created directory: {new_dir_name}")
        os.chdir(new_dir_name)

        #############################

    # atari-3 : Battle Zone, Name This Game, Phoenix
    # atari-5 : Battle Zone, Double Dunk, Name This Game, Phoenix, Q*Bert

    if testing:
        num_envs = 4
        eval_envs = 3
        eval_every = 30000
        num_eval_episodes = 10
        n_steps = 25000
        bs = 16
    else:
        num_envs = envs
        eval_envs = 32
        n_steps = frames
        num_eval_episodes = 100
        eval_every = 1000000

    next_eval = eval_every

    print("Currently Playing Game: " + str(game))

    gpu = "0"
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    env = make_env(num_envs)

    print(env.observation_space)
    print(env.action_space[0])

    agent = Agent(n_actions=env.action_space[0].n, input_dims=[4, 84, 84], device=device, num_envs=num_envs,
                  agent_name=agent_name, total_frames=n_steps, testing=testing, batch_size=bs, rr=rr, lr=lr,
                  maxpool_size=maxpool_size, ema=ema, trust_regions=tr, target_replace=c, ema_tau=ema_tau,
                  noisy=noisy, spectral=spectral, munch=munch, iqn=iqn, double=double, dueling=dueling, impala=impala,
                  discount=discount, adamw=adamw, ede=ede, sqrt=sqrt, discount_anneal=discount_anneal, lr_decay=lr_decay,
                  per=per, taus=taus, moe=moe, pruning=pruning, model_size=model_size, linear_size=linear_size)

    if wandb_logs:
        wandb.init(
            # set the wandb project where this run will be logged
            project="BeyondTheRainbow",
            save_code=True,
            name="Trial",

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

    evals_total = []

    scores_count = [0 for i in range(num_envs)]

    dormants = []
    param_norms = []

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
                if wandb_logs:
                    wandb.log({"train_scores": scores_count[i], "steps": steps, "episodes": episodes,
                               "walltime": time.time() - start})

                scores_count[i] = 0

        reward = np.clip(reward, -1., 1.)

        for stream in range(num_envs):
            agent.store_transition(observation[stream], action[stream], reward[stream], done_[stream], stream=stream)

        observation = observation_

        for stream in range(num_envs):
            if done_[stream]:
                observation[stream] = info["final_observation"][stream]

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

            dormants.append(agent.get_dormant_neurons())

            # get and save percent of dormant neurons
            fname = agent_name + "Dormants.npy"
            if not testing:
                np.save(fname, np.array(dormants))

            # get Parameter Norm
            param_norms.append(agent.calculate_parameter_norms())
            fname = agent_name + "ParamNorms.npy"
            if not testing:
                np.save(fname, np.array(param_norms))

            fname = agent_name + "Experiment.npy"
            if not testing:
                np.save(fname, np.array(scores))

            eval_env = make_env(eval_envs)

            agent.set_eval_mode()
            evals = []
            eval_episodes = 0
            eval_scores = np.array([0 for i in range(eval_envs)])
            eval_observation, eval_info = eval_env.reset()

            evals_started = [i for i in range(eval_envs)]

            while eval_episodes < num_eval_episodes:

                eval_action = agent.choose_action(eval_observation)  # this takes and return batches

                eval_observation_, eval_reward, eval_done_, eval_trun_, eval_info = eval_env.step(eval_action)

                # TRUNCATION NOT IMPLEMENTED
                eval_done_ = np.logical_or(eval_done_, eval_trun_)

                for i in range(eval_envs):
                    eval_scores[i] += eval_reward[i]

                    if eval_done_[i]:
                        if evals_started[i] < num_eval_episodes:
                            evals_started[i] = max(evals_started) + 1
                            eval_episodes += 1
                            if wandb_logs:
                                wandb.log({"eval_scores": eval_scores[i]})
                            evals.append(eval_scores[i])

                            eval_scores[i] = 0

                    if len(evals) == num_eval_episodes:
                        break

                eval_observation = eval_observation_

                for stream in range(eval_envs):
                    if eval_done_[stream]:
                        eval_observation[stream] = eval_info["final_observation"][stream]

            evals_total.append(evals)
            fname = agent_name + game + "Evaluation.npy"
            if not testing:
                np.save(fname, np.array(evals_total))
            next_eval += eval_every
            agent.set_train_mode()

    agent.save_model()
    if wandb_logs:
        wandb.finish()
