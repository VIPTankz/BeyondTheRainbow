import torch
import numpy as np
import itertools
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

        self.input_shape = input_shape

        self.framestack = self.input_shape[0]

        self.idxs = np.array([0 for i in range(self.num_envs)])
        # these are the idxs in the buffer we are at relative to the start of each chunk (NOT absolute)
        self.max_idx = size // num_envs

        self.max_per_env = np.array([-1 for i in range(self.num_envs)])  # this basically just tells us the highest we have reached
        # this is used to know where we can sample from
        self.total = 0

        self.invalids = np.array([[] for i in range(self.num_envs)], dtype=int)

        self.full = False

        # the [1:] removes the framestack dim
        self.state_mem = torch.empty((self.size, *input_shape[1:]), dtype=torch.uint8)
        self.action_mem = torch.empty(self.size, dtype=torch.int64)
        self.reward_mem = torch.empty(self.size, dtype=torch.float32)
        self.terminal_mem = torch.empty(self.size, dtype=torch.bool)
        self.not_samplable = []

    def append(self, state, action, reward, terminal, stream, samplable=True):

        idx = self.idxs[stream] + stream * self.max_idx

        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.terminal_mem[idx] = terminal

        if not samplable:
            for i in range(self.n + self.framestack):
                self.not_samplable.append((idx - i) % self.max_idx)

        self.idxs[stream] = (self.idxs[stream] + 1) % self.max_idx
        self.total += 1
        self.max_per_env[stream] = min(self.max_per_env[stream] + 1, self.max_idx)

        self.invalids = self.create_invalid_sequences(self.idxs)

        if not self.full:
            full = True
            for i in self.max_per_env:
                if i != self.max_idx:
                    full = False
                    break

            if full:
                self.full = True

    def batch_append(self, states, actions, rewards, terminals):
        states = states[self.input_shape[0] - 1, :, :]

        # gets the idxs where the new transitions will be placed
        new_idxs = np.arange(self.num_envs) * self.max_idx + self.idxs

        self.state_mem[new_idxs] = states.to(dtype=torch.uint8, device=torch.device('cpu'))
        self.action_mem[new_idxs] = actions.to(dtype=torch.int64)
        self.reward_mem[new_idxs] = rewards.to(dtype=torch.float32)
        self.terminal_mem[new_idxs] = terminals.to(dtype=torch.bool)
        # could potentially be added later though

        # update which idxs are valid to be sampled from (N-step)
        self.invalids = self.create_invalid_sequences(self.idxs)

        # update idxs and stats for next time
        self.idxs += 1
        self.idxs = self.idxs % self.max_idx
        self.total += self.num_envs
        self.max_per_env = np.minimum(self.max_per_env + 1, self.max_idx)

        # since batches have same amount for each env, can just check one
        if not self.full:
            if self.max_per_env[0] == self.max_idx:
                self.full = True

    def create_invalid_sequences(self, arr):
        # Create an array of offsets
        offsets = np.arange(self.framestack + self.n, -1, -1)

        # Subtract offsets from each element in arr (broadcasting)
        sequences = (arr.reshape(-1, 1) - offsets) % self.max_idx

        if len(self.not_samplable) > 0:
            # add anything which should not be sampled
            sequences = np.append(sequences, np.array(self.not_samplable))

        return sequences

    def generate_idxs(self, bs):
        if self.full:
            return np.random.randint(0, self.size + 1, bs)
        else:
            # this technically isn't uniform random if envs have different nums of transitions
            # this basically randomly chooses which env to look at, then samples a random transition from that env

            indices = np.random.randint(low=0, high=self.num_envs, size=bs)

            selected_limits = np.array([self.max_per_env[i] for i in indices])

            return np.array([np.random.randint(0, limit) for limit in selected_limits])


    def sample(self, bs):
        # This will break if self.total < bs!
        # please do your checks somewhere else!

        # Need to decide the idxs we are sampling from

        idxs = self.generate_idxs(bs)
        # this gets indices from anywhere, but we still need to check they aren't invalid due to N-step

        first_mask = np.ones(bs, dtype=bool)

        while True:
            # Check if each element in data is in the invalid list
            new_mask = ~np.isin(idxs, self.invalids)

            # Check if there were any changes compared to the last iteration
            if np.array_equal(new_mask, first_mask):
                break  # Exit the loop if no changes

            # Update the data and mask for the next iteration
            mask = ~new_mask
            idxs[mask] = self.generate_idxs(np.sum(mask))

        # do masks over memories

        actions = self.action_mem[idxs]
        states, rewards, next_states, terminals = self.lookahead(idxs, bs)
        states = states.to(torch.float32).cuda()
        next_states = next_states.to(torch.float32).cuda()

        nonterminals = ~terminals

        return states, actions, rewards, next_states, nonterminals

    def lookahead(self, idxs, bs):
        # this doesn't handle edge of range!
        states = self.state_mem[idxs - self.framestack:idxs]
        next_states = self.state_mem[idxs + self.n - self.framestack:idxs + self.n]

        terminals = self.terminal_mem[idxs + self.n - self.framestack:idxs - self.framestack + self.n]
        terminals = terminals.any(dim=1)

        # this one has to look forward and apply gamma
        rewards = self.reward_mem[idxs]  # to start with

        temp_terminals = self.terminal_mem[idxs]

        return states, rewards, next_states, terminals




if __name__ == "__main__":
    er = ExperienceReplay(20, 2, 3, 0.99, [1, 1])
    er.batch_append(torch.tensor([[[5], [6]], [[2], [3]]]), torch.tensor([1, 2]),
                    torch.tensor([0.2, 0.3]), torch.tensor([False, False]))


