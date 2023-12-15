with open('logg.txt', 'w') as f:
    f.write("Created Fresh Log File")


import sys
import os

from multiprocessing.connection import Listener,Client
import numpy as np
from PIL import Image, ImageEnhance
from collections import deque
import random
import time
import gymnasium

with open('logg.txt', 'a') as f:
    f.write("Imported Libs")

with open('pid_num.txt') as f:
    pid = int(f.readlines()[0])

with open('script_pid' + str(pid) + '.txt', 'w') as f:
    f.write(str(os.getpid()))

with open('game.txt') as f:
    game = f.readlines()[0]

def make_env(game, eval):
    env = gymnasium.make('ALE/' + game + '-v5', frameskip=1)


    env = gymnasium.wrappers.AtariPreprocessing(env)
    env = gymnasium.wrappers.FrameStack(env, 4)

    return env


class DolphinEnv:
    def __init__(self):

        self.env = make_env(game, False)

        addressClient = ('localhost', 26330 + pid)
        self.c_conn = Client(addressClient, authkey=b'secret password')
        self.c_conn.send("Start, from Scripter")

        addressListener = ('localhost', 25330 + pid)
        listener = Listener(addressListener, authkey=b'secret password')
        self.l_conn = listener.accept()
        msg = self.l_conn.recv()

    def send_init_state(self, img):
        self.c_conn.send(img)

    def send_transition(self, state, action, reward, terminal, new_img):

        self.c_conn.send([state, action, reward, terminal, new_img])

    def reset(self):
        state_, _ = self.env.reset()
        state_ = state_._frames

        return np.array(state_)


    def action(self):
        # code to fetch action

        running = True
        while running:
            while self.l_conn.poll():
                action = self.l_conn.recv()
                running = False


        # apply the action
        self.save_action = action

        return action

    def step(self):

        observation_, reward, done_, truncated, info = self.env.step(self.save_action)
        done_ = done_ or truncated

        observation_ = np.array(observation_._frames)

        return observation_, reward, done_


env = DolphinEnv()
img = env.reset()

env.send_init_state(img)

reward = 0
terminal = False

while True:

    action = env.action()

    new_img, reward, terminal = env.step()

    env.send_transition(img, action, reward, terminal, new_img)

    img = new_img

    if terminal:
        img = env.reset()

    reward = 0
    terminal = False
