import torch
import numpy as np

"""
Implementation of regular Experience Replay

Supports:
VectorInput
Memory Efficient (stores each frame once)
N-Step
Capable of changing N and gamma
Fast!
Can also set transitions to NOT be sampled (This helps with applications where things can crash)
"""

class ExperienceReplay:
    def __init__(self, size, num_envs, n, gamma, input_shape):
        self.size = size
        self.num_envs = num_envs  # please make sure this is a power two if possible (although technically just
        # needs to be divisible by num_envs)

        self.n = n
        self.gamma = gamma

        self.idxs = np.array([0 for i in range(self.num_envs)])
        # these are the idxs in the buffer we are at relative to the start of each chunk (NOT absolute)
        self.max_idx = size // num_envs

        self.max_per_env = -1  # this basically just tells us the highest we have reached
        # this is used to know where we can sample from
        self.total = 0

        # the [1:] removes the framestack dim
        self.state_mem = torch.empty((self.size, *input_shape[1:]), dtype=torch.uint8)
        self.action_mem = torch.empty(self.size, dtype=torch.uint8)
        self.reward_mem = torch.empty(self.size, dtype=torch.float32)
        self.terminal_mem = torch.empty(self.size, dtype=torch.bool)
        self.samplable_mem = torch.empty(self.size, dtype=torch.bool)

    def append(self, state, action, reward, terminal, stream, samplable=True):

        idx = self.idxs[stream] + stream * self.max_idx

        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.terminal_mem[idx] = terminal
        self.samplable_mem[idx] = samplable  # this is not an option for batch input

        self.idxs[stream] = (self.idxs[stream] + 1) % self.max_idx
        self.total += 1
        self.max_per_env[stream] = min(self.max_per_env[stream] + 1, self.max_idx)

    def batch_append(self, states, actions, rewards, terminals):

        # gets the idxs where the new transitions will be placed
        new_idxs = np.arange(self.num_envs) * self.max_idx + self.idxs

        self.state_mem[new_idxs] = states
        self.action_mem[new_idxs] = actions
        self.reward_mem[new_idxs] = rewards
        self.terminal_mem[new_idxs] = terminals
        self.samplable_mem[new_idxs] = torch.ones_like(terminals)  # this is not an option for batch input
        # could potentially be added later though

        # update idxs and stats for next time
        self.idxs += 1
        self.idxs = self.idxs % self.max_idx
        self.total += self.num_envs
        self.max_per_env = min(self.max_per_env + 1, self.max_idx)


    def sample(self, bs):
        pass


if __name__ == "__main__":
    er = ExperienceReplay(20, 2, 3, 0.99, [1, 1])
    er.batch_append([[[5], [6]], [[2], [3]]], [1, 2], [0.2, 0.3], [False, False])


