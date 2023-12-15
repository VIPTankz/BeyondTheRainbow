with open('logg.txt', 'w') as f:
    f.write("Created Fresh Log File")

from dolphin import event, gui, savestate, memory, controller
from multiprocessing.connection import Listener,Client

import sys
import os
sys.path.append("C:\\Users\\Tyler\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages")

from multiprocessing.connection import Listener,Client
import numpy as np
from PIL import Image, ImageEnhance
from collections import deque
import random
import time

with open('logg.txt', 'a') as f:
    f.write("Imported Libs")

with open('pid_num.txt') as f:
    pid = int(f.readlines()[0])

with open('script_pid' + str(pid) + '.txt', 'w') as f:
    f.write(str(os.getpid()))


class DolphinEnv:
    def __init__(self):
        self.frameskip = 4
        self.framestack = 4
        self.frames = deque([], maxlen=self.framestack) #frame buffer for framestacking
        self.window_x = 140
        self.window_y = 75

        addressClient = ('localhost', 26330 + pid)
        self.c_conn = Client(addressClient, authkey=b'secret password')
        self.c_conn.send("Start, from Scripter")

        addressListener = ('localhost', 25330 + pid)
        listener = Listener(addressListener, authkey=b'secret password')
        self.l_conn = listener.accept()
        msg = self.l_conn.recv()

        self.reset()

    def send_init_state(self, img):
        self.c_conn.send(img)

    def process_frame(self, image, terminal):
        # internetal res is 640x528
        # at internet res, frame dump gives 832x456

        image = image.convert("L")
        #image = image.crop((64,16,768,392))

        image = image.resize((self.window_x, self.window_y))

        image = np.asarray(image)

        #process Uint
        image = image.astype(np.uint8)
        image = np.expand_dims(image, axis=2)

        #Convert to PyTorch Image
        image = np.swapaxes(image,2,0)

        #Framestack
        if terminal or len(self.frames) != self.framestack:
            for i in range(self.framestack):
                self.frames.append(image) #self.frames has max length 4, so this just resets it
        else:
            self.frames.append(image)

        observation = np.array(self.frames).squeeze()

        return observation

    def send_transition(self, state, action, reward, terminal, new_img):

        self.c_conn.send([state, action, reward, terminal, new_img])


    def action(self):
        # code to fetch action

        running = True
        while running:
            while self.l_conn.poll():
                action = self.l_conn.recv()
                running = False


        # apply the action
        self.apply_action(action)

        return action

    def reset(self):

        self.ep_length = 0
        self.is_terminal = False
        self.last_lives = -1
        self.last_action = 0

        self.last_value = -1
        self.section = 0

        self.last_dir = 0

        savestate.load_from_slot(1)

        self.cur_dir = 0
        self.dir_frames = 0

        self.wii_dic = {"A": False,
                "B": False,
                "One": False,
                "Two": False,
                "Plus": False,
                "Minus": False,
                "Home": False,
                "Up": False,
                "Down": False,
                "Left": False,
                "Right": False}
        self.nun_dic = {"C": False,
                "Z": False,
                "StickX": 0,
                "StickY": 0}

        ##################### End Game Code


    def apply_action(self, action=None):
        if action is None:
            action = self.last_action
        else:
            self.last_action = action

        self.wii_dic = {"A": False,
                "B": False,
                "One": False,
                "Two": False,
                "Plus": False,
                "Minus": False,
                "Home": False,
                "Up": False,
                "Down": False,
                "Left": False,
                "Right": False}
        self.nun_dic = {"C": False,
                "Z": False,
                "StickX": self.nun_dic["StickX"],
                "StickY": self.nun_dic["StickY"]}

        if action == 0:
            self.nun_dic["StickX"] = 0.0
            self.nun_dic["StickY"] = 0.0
        if action == 1:
            self.nun_dic["StickX"] = 1.0
            self.nun_dic["StickY"] = 0.0
        elif action == 2:
            self.nun_dic["StickX"] = -1.0
            self.nun_dic["StickY"] = 0.0
        elif action == 3:
            self.nun_dic["StickY"] = 1.0
            self.nun_dic["StickX"] = 0.0
        elif action == 4:
            self.nun_dic["StickY"] = -1.0
            self.nun_dic["StickX"] = 0.0
        elif action == 5:
            self.wii_dic["A"] = True

        controller.set_wii_nunchuk_buttons(0, self.nun_dic)
        controller.set_wiimote_buttons(0, self.wii_dic)

    def get_reward_terminal(self):
        # Returns reward,terminal

        terminal = False
        reward = 0

        lives = memory.read_u8(0x81064CD3)
        total_lives = memory.read_u8(0x80F63CF1)

        #FIRST SECTION
        coord1 = memory.read_f32(0x8114958C)

        #5112 is where section ends

        #SECOND SECTION
        coord2 = memory.read_f32(0x8087B1B8)

        random_var = memory.read_f32(0x8105D4DC)

        finished = memory.read_u8(0x807973ED)
        #0 for not finished, 1 when finished

        if self.section == 0 and self.last_value != -1:
            reward += (coord1 - self.last_value) / 150

            if 22.9 < random_var < 24.1:
                reward += 1
                self.section = 1
                self.last_value = -1

        if self.section == 1 and self.last_value != -1:
            reward += (coord2 - self.last_value) / 150

            if 24.1 < random_var < 26.0 or 23.1 < random_var < 23.8:
                reward -= 1
                self.section = 0
                self.last_value = -1

        if self.section == 0:
            self.last_value = coord1
        else:
            self.last_value = coord2

        if self.last_lives != -1:
            reward -= (self.last_lives - lives)

        self.last_lives = lives

        if total_lives == 3:
            return -1, True


        if finished and self.section == 1 and coord2 > 25250 and coord2 < 25750:
            reward = 10
            terminal = True

        # failsafe
        if reward > 20 or reward < -20:
            return -1.0, True

        return reward, terminal


for i in range(4):
    await event.frameadvance()

env = DolphinEnv()


for i in range(env.frameskip):
    await event.frameadvance()


def my_callback():
    env.apply_action()

event.on_frameadvance(my_callback)

(width,height,data) = await event.framedrawn()
img = Image.frombytes('RGBA', (width,height), data, 'raw')

img = env.process_frame(img, terminal=False)
env.send_init_state(img)

red = 0xffff0000
reward = 0
terminal = False

with open('logg.txt', 'a') as f:
    f.write("\nBefore Main Loop")

while True:

    action = env.action()

    for i in range(env.frameskip):
        (width,height,data) = await event.framedrawn()

        rewardN,terminalN = env.get_reward_terminal()

        if not terminal:
            terminal = terminal or terminalN
            reward += rewardN

        if terminal:
            for i in range(2):
                await event.frameadvance()

            env.reset()

            for i in range(2):
                await event.frameadvance()
            (width,height,data) = await event.framedrawn()
            break

    new_img = Image.frombytes('RGBA', (width,height), data, 'raw')
    new_img = env.process_frame(new_img, terminal=terminal)

    env.send_transition(img, action, reward, terminal, new_img)

    #check this to make sure deepcopy isn't needed
    img = new_img

    reward = 0
    terminal = False
