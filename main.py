
import numpy as np
import time
from copy import deepcopy
import sys
import torch
import gymnasium as gym


def make_env(envs_create):
    return gym.vector.AsyncVectorEnv([lambda: gym.wrappers.FrameStack(
        gym.wrappers.AtariPreprocessing(gym.make("ALE/" + game + "-v5", frameskip=1)), 4) for _ in range(envs_create)])


if __name__ == '__main__':

    from Agent import Agent

    agent_name = "BTR_Trial"

    # atari-3 : Battle Zone, Name This Game, Phoenix
    # atari-5 : Battle Zone, Double Dunk, Name This Game, Phoenix, Q*Bert

    num_envs = 1
    n_steps = 3000 #50000000
    gameset = ["UpNDown"]

    print("Currently Playing Game(s): " + str(gameset))

    gpu = sys.argv[1]
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    for game in gameset:

        # gym version 0.25.2
        # ie pre 5 arg step

        env = make_env(num_envs)

        print(env.observation_space)
        print(env.action_space[0])

        agent = Agent(n_actions=env.action_space[0].n, input_dims=[4, 84, 84], device=device, num_envs=num_envs,
                      agent_name=agent_name, total_frames=n_steps)

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
            # need to sort out new API

            # this is placed here so learning takes place while step is happening
            agent.learn()

            observation_, reward, done_, trun_, info = env.step_wait()

            #TRUNCATATION NOT IMPLEMENTED
            done_ = np.logical_or(done_, trun_)

            for i in range(num_envs):
                scores_count[i] += reward[i]

                if done_[i]:
                    scores.append([scores_count[i], steps])
                    scores_temp.append(scores_count[i])
                    scores_count[i] = 0

            reward = np.clip(reward, -1., 1.)

            for stream in range(num_envs):
                agent.store_transition(observation[stream], action[stream], reward[stream], done_[stream], stream=stream)

            observation = deepcopy(observation_)

            if steps % 1200 == 0:

                avg_score = np.mean(scores_temp[-50:])

                if episodes % 1 == 0:
                    print('{} {} avg score {:.2f} total_steps {:.0f} fps {:.2f}'
                          .format(agent_name, game, avg_score, steps, steps / (time.time() - start)), flush=True)

        fname = agent_name + game + "Experiment.npy"
        np.save(fname, np.array(scores))

        env = make_env(1)

        agent.set_eval_mode()
        evals = []
        steps = 0
        eval_episodes = 0
        while eval_episodes < 100:
            done = False
            observation, _ = env.reset()
            observation = observation
            score = 0
            while not done:
                steps += 1
                action = agent.choose_action(observation)

                observation_, reward, done_, trun_, info = env.step(action)
                observation_ = observation_

                # TRUNCATATION NOT IMPLEMENTED
                done = np.logical_or(done_, trun_)
                reward = reward[0]

                score += reward
                observation = observation_

            evals.append(score)
            print("Evaluation Score: " + str(score))
            eval_episodes += 1
            print("Average:")
            print(np.mean(np.array(evals)))

        fname = agent_name + game + "Evaluation.npy"
        np.save(fname, np.array(evals))
