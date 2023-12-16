import os
import time
from multiprocessing.connection import Listener,Client
import gymnasium
import numpy as np
from Agent import Agent
from collections import deque
from copy import deepcopy
import subprocess
import torch
class Communicator:
    def __init__(self, envs, agent, batch_frames, run_name):
        self.num_envs = envs
        self.agent = agent
        self.batch_frames = batch_frames
        self.run_name = run_name

        self.script_pids = [None for i in range(self.num_envs)]

        self.l_conns = [None for i in range(self.num_envs)]
        self.c_conns = [None for i in range(self.num_envs)]

        self.dead_frames = [0 for i in range(self.num_envs)]
        self.dead_threshold = 250000
        self.save_frames = [None for i in range(self.num_envs)]

        self.last_frames = [None for i in range(self.num_envs)]
        self.transitions = deque([])

        self.state_size_x = 140
        self.state_size_y = 75

        self.total_env_frames = 0
        self.total_grad_frames = 0
        self.start = time.time()
        self.inst_time = time.time()

        self.total_rewards = []
        self.reward_tallies = [0 for i in range(self.num_envs)]

        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(4, self.state_size_y, self.state_size_x), dtype=np.uint8)
        self.action_space = gymnasium.spaces.Discrete(4)

        with open('pid_num.txt', 'w') as f:
            f.write("0")

        for i in range(self.num_envs):
            self.create_dolphin(i)

    def create_dolphin(self, pid):
        with open('pid_num.txt', 'w') as f:
            f.write(str(pid))

        time.sleep(0.1)
        print("Launching... ")
        self.launch(pid)
        print("Launched Successfully")

        addressListener = ('localhost', 26330 + pid)
        listener = Listener(addressListener, authkey=b'secret password')

        print("Waiting for listener")
        self.l_conns[pid] = listener.accept()
        msg = self.l_conns[pid].recv()
        print("Listener Created Successfully")

        addressClient = ('localhost', 25330 + pid)
        self.c_conns[pid] = Client(addressClient, authkey=b'secret password')
        time.sleep(.5)
        self.c_conns[pid].send("Start, from Main Env")
        print("Created Client Successfully")

        # receive initial states
        self.last_frames[pid] = self.l_conns[pid].recv()
        self.save_frames[pid] = deepcopy(self.last_frames[pid])
        

    def launch(self, pid):

        # launch dolphin
        string = "python DolphinScript.py"
        os.popen(string)

        time.sleep(1)

        if not os.path.isfile('script_pid' + str(pid) + '.txt'):
            with open('script_pid' + str(pid) + '.txt', 'w') as f:
                f.write('0')

        with open('script_pid' + str(pid) + '.txt') as f:
            self.script_pids[pid] = int(f.readlines()[0])

        time.sleep(0.5)

    def send_actions(self):
        #this could potentially be batched

        for i in range(len(self.last_frames)):
            if self.last_frames[i] is not None:
                action = self.agent.choose_action(self.last_frames[i], i)
                self.c_conns[i].send(action)
                self.last_frames[i] = None

    def send_actions_human(self, action):
        #this could potentially be batched

        for i in range(len(self.last_frames)):
            if self.last_frames[i] is not None:
                self.c_conns[i].send(action)
                self.last_frames[i] = None

    def learn(self):
        while len(self.transitions) >= com.batch_frames:
            for i in range(com.batch_frames):

                if self.total_env_frames % 1600 == 0:

                    print_str = "Frames: " + str(self.total_env_frames)

                    print_str += ", Recent FPS: " + str(round(1600 / (time.time() - self.inst_time),2))
                    print_str += ", Total FPS: " + str(round(self.total_env_frames / (time.time() - self.start),2))

                    if len(self.total_rewards) > 1:

                        print_str += ", Avg Reward: " + str(round(float(np.array(self.total_rewards[-100:]).mean(axis=0)[0]), 2))

                    print_str += ", Time: " + str(round(round(time.time() - self.start, 2) / 3600, 2)) + " Hours"

                    print(print_str)

                    self.inst_time = time.time()

                if self.total_env_frames % 25000 == 0:
                    np.save(self.run_name, self.total_rewards)

                if self.total_env_frames % SAVE_INTERVAL == 0:
                    self.agent.save_model()

                transition = self.transitions.pop()

                """ Testing Stuff
                print(transition[0].shape)
                im = Image.fromarray(transition[0][0])
                im.save("test_state.jpeg")
                raise Exception("stop")
                """

                state, action, reward, terminal, new_img, stream = transition
                self.agent.store_transition(state, action, reward, terminal, stream)
                self.total_env_frames += 1

            self.agent.learn()

    def detect_fault(self, i):
        self.dead_frames[i] += 1

        if self.dead_frames[i] > self.dead_threshold:
            #kill and restart
            print("Detected Crash!")
            try:
                subprocess.check_output("Taskkill /PID %d /F" % self.script_pids[i])
            except:
                print("Full Crash!")
            time.sleep(0.02)

            msg = [self.save_frames[i], 0, 0, True, self.save_frames[i], i]
            self.transitions.appendleft(msg)

            time.sleep(0.02)
            self.create_dolphin(i)
            self.dead_frames[i] = 0

    def scan_frames(self):
        for i in range(self.num_envs):

            #this needs to be non-blocking. Also needs to check for crashes.
            if self.l_conns[i].poll():
                try:
                    msg = self.l_conns[i].recv()
                    msg.append(i)
                    self.dead_frames[i] = 0
                except:
                    self.detect_fault(i)
                    continue
            else:
                self.detect_fault(i)
                continue

            #only continue here if we actually got a frame


            #here turn this into actual transition
            # (state, action, reward t+1, terminal t+1, state t+1)
            self.last_frames[i] = deepcopy(msg[-2])
            self.save_frames[i] = deepcopy(msg[-2])

            self.transitions.appendleft(msg)

            #add to reward totals
            self.reward_tallies[i] += msg[2]

            if self.agent is None:
                print("Reward: " + str(msg[2] / 4))

            if msg[3]:
                if self.agent is None:
                    print("Total Reward: " + str(self.reward_tallies[i]))
                self.total_rewards.append([self.reward_tallies[i], self.total_env_frames, time.time() - self.start])
                self.reward_tallies[i] = 0


def on_press(key):
    global action
    try:
        if key.char == '1':
            action = 1
        elif key.char == '2':
            action = 2
        elif key.char == '3':
            action = 3
        elif key.char == '4':
            action = 4
        elif key.char == '5':
            action = 5
        elif key.char == '6':
            action = 6
        elif key.char == '7':
            action = 7
        elif key.char == '8':
            action = 8
        elif key.char == '9':
            action = 9

    except:pass
        
def on_release(key):
    global action
    action = 0


def make_env(game, eval=False):
    env = gymnasium.make('ALE/' + game + '-v5', frameskip=1)

    env = gymnasium.wrappers.AtariPreprocessing(env)
    env = gymnasium.wrappers.FrameStack(env, 4)

    return env

if __name__ == "__main__":

    game = "NameThisGame"
    agent_name = "RainbowImpala2SpectralMaxPoolMunchausenIqnNoNoisyBs16_" + game

    #target replace is every 8000 grad / 32000 env steps

    env = make_env(game)

    with open('game.txt', 'w') as f:
        f.write(game)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    envs = 4
    replay_period = 4
    SAVE_INTERVAL = 1000000

    action = 0

    mode = "ai"

    if mode == "ai":
        agent = Agent(env.action_space.n, [4, 84, 84], device=device, num_envs=envs, agent_name=agent_name)

        com = Communicator(envs=envs, agent=agent, batch_frames=replay_period, run_name=agent_name)
    else:
        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
        listener.start()
        com = Communicator(envs=1, agent=None, batch_frames=8, run_name="Test")

    if mode == "human":
        while True:
            time.sleep(0.01)
            com.send_actions_human(action)
            com.scan_frames()


    while True:
        com.send_actions()
        com.learn()
        com.scan_frames()

