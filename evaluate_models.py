import numpy as np
import torch
import gymnasium as gym
import os
import argparse
from Agent import Agent

def make_env(envs_create):
    return gym.vector.SyncVectorEnv([lambda: gym.wrappers.FrameStack(
        gym.wrappers.AtariPreprocessing(gym.make("ALE/" + game + "-v5", frameskip=1)), 4) for _ in
                                     range(envs_create)])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default="BattleZone")
    parser.add_argument('--envs', type=int, default=64)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--rr', type=int, default=1)
    parser.add_argument('--frames', type=int, default=50000000)
    parser.add_argument('--evals', type=int, default=200)
    parser.add_argument('--agent_name', type=str, default="FinalAgent")

    parser.add_argument('--maxpool_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--ema_tau', type=float, default=2.5e-4)
    parser.add_argument('--munch', type=int, default=1)

    # the way parser.add_argument handles bools in dumb so we use int 0 or 1 instead
    parser.add_argument('--noisy', type=int, default=1)
    parser.add_argument('--spectral', type=int, default=1)
    parser.add_argument('--spectral_lin', type=int, default=0)
    parser.add_argument('--iqn', type=int, default=1)

    parser.add_argument('--impala', type=int, default=1)
    parser.add_argument('--discount', type=float, default=0.997)
    parser.add_argument('--adamw', type=int, default=1)
    parser.add_argument('--lr_decay', type=int, default=0)
    parser.add_argument('--per', type=int, default=1)
    parser.add_argument('--taus', type=int, default=8)
    parser.add_argument('--c', type=int, default=500)  # this is the target replace
    parser.add_argument('--dueling', type=int, default=1)

    # features still in testing

    parser.add_argument('--linear_size', type=int, default=512)
    parser.add_argument('--model_size', type=int, default=2)
    parser.add_argument('--tr', type=int, default=0)
    parser.add_argument('--ncos', type=int, default=64)

    # not applicable when using munchausen
    parser.add_argument('--double', type=int, default=0)

    # likely dead improvements
    parser.add_argument('--sqrt', type=int, default=0)
    parser.add_argument('--ede', type=int, default=0)
    parser.add_argument('--discount_anneal', type=int, default=0)
    parser.add_argument('--moe', type=int, default=0)  # This Does not Work Yet!
    parser.add_argument('--pruning', type=int, default=0)  # ONLY WORKS FOR DUELING
    parser.add_argument('--ema', type=int, default=0)

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
    spectral_lin = args.spectral_lin
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
    ncos = args.ncos
    num_evals = args.evals

    discount_anneal = args.discount_anneal

    # atari-3 : Battle Zone, Name This Game, Phoenix
    # atari-5 : Battle Zone, Double Dunk, Name This Game, Phoenix, Qbert

    lr_str = "{:e}".format(lr)
    lr_str = str(lr_str).replace(".", "").replace("0", "")
    frame_name = str(int(frames / 1000000)) + "M"

    include_evals = False

    # python .\evaluation.py --testing 1 --evals 40 --game NameThisGame --name FullAgent
    name_ending = args.agent_name
    agent_name = "BTR_" + game + frame_name + "_" + name_ending

    print("Agent Name:" + str(agent_name))
    testing = args.testing
    wandb_logs = False

    # atari-3 : Battle Zone, Name This Game, Phoenix
    # atari-5 : Battle Zone, Double Dunk, Name This Game, Phoenix, Q*Bert

    if testing:
        eval_envs = 4
        num_eval_episodes = 5
    else:
        eval_envs = 25
        num_eval_episodes = 100

    print("Currently Playing Game: " + str(game))

    gpu = "0"
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    print("Eval Envs: " + str(eval_envs))

    eval_env = make_env(eval_envs)

    print(eval_env.observation_space)
    print(eval_env.action_space[0])

    agent = Agent(n_actions=eval_env.action_space[0].n, input_dims=[4, 84, 84], device=device, num_envs=eval_envs,
                  agent_name=agent_name, total_frames=50000000, testing=testing, batch_size=bs, rr=rr, lr=lr,
                  maxpool_size=maxpool_size, ema=ema, trust_regions=tr, target_replace=c, ema_tau=ema_tau,
                  noisy=noisy, spectral=spectral, munch=munch, iqn=iqn, double=double, dueling=dueling, impala=impala,
                  discount=discount, adamw=adamw, ede=ede, sqrt=sqrt, discount_anneal=discount_anneal, lr_decay=lr_decay,
                  per=per, taus=taus, moe=moe, pruning=pruning, model_size=model_size, linear_size=linear_size,
                  spectral_lin=spectral_lin, ncos=ncos)

    evals_total = []

    os.chdir(agent_name)

    for evaluation in range(num_evals):

        evaluation += 1
        print("Starting Evaluation " + str(evaluation) + "M")
        print(agent_name + "_" + str(evaluation) + "M.model")

        agent.load_models(agent_name + "_" + str(evaluation) + "M.model")

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
                        evals.append(eval_scores[i])

                        eval_scores[i] = 0

                if len(evals) == num_eval_episodes:
                    break

            eval_observation = eval_observation_

            for stream in range(eval_envs):
                if eval_done_[stream]:
                    eval_observation[stream] = eval_info["final_observation"][stream]

        print(np.mean(evals))
        evals_total.append(evals)
        fname = "BTR_" + game + str(num_evals // 4) + "M_" + name_ending + "Evaluation.npy"
        np.save(fname, np.array(evals_total))

