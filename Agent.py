import os
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memory import ReplayMemory
import numpy as np
from collections import deque
import pickle
import matplotlib.pyplot as plt
#from torchsummary import summary
import math
from networks import ImpalaCNNLarge, ImpalaCNNLargeIQN, NatureIQN
import traceback

class EpsilonGreedy():
    def __init__(self, eps_start, eps_steps, eps_final, action_space):
        self.eps = eps_start
        self.steps = eps_steps
        self.eps_final = eps_final
        self.action_space = action_space

    def update_eps(self):
        self.eps = max(self.eps - (self.eps - self.eps_final) / self.steps, self.eps_final)

    def choose_action(self):
        if np.random.random() > self.eps:
            return None
        else:
            return np.random.choice(self.action_space)


class Agent():
    def __init__(self, n_actions, input_dims, device, num_envs, agent_name, total_frames, testing=False, batch_size=16
                 , rr=1, maxpool_size=6, lr=5e-5):

        self.n_actions = n_actions
        self.input_dims = input_dims
        self.device = device
        self.agent_name = agent_name
        self.testing = testing

        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.chkpt_dir = ""

        # IMPORTANT params, check these

        if self.testing:
            self.min_sampling_size = 8000
            self.lr = 0.0001
        else:
            self.min_sampling_size = 200000
            self.lr = lr

        self.n = 3
        self.gamma = 0.99
        self.batch_size = batch_size

        self.replay_ratio = rr
        self.model_size = 2  # Scaling of IMPALA network
        self.maxpool_size = maxpool_size

        # do not use both spectral and noisy, they will interfere with each other
        self.noisy = False
        self.spectral_norm = True  # this produces nans for some reason! - using torch.autocast('cuda') fixed it?
        # RIP mental sanity

        self.per_splits = 1
        if self.per_splits > num_envs:
            self.per_splits = num_envs

        self.impala = True #non impala only implemented for iqn

        # Don't use both of these, they are mutually exclusive
        self.c51 = False
        self.iqn = True

        self.double = False  # Not implemented for IQN and Munchausen
        self.maxpool = True
        self.munchausen = True

        if self.munchausen:
            self.entropy_tau = 0.03
            self.lo = -1
            self.alpha = 0.9

        self.max_mem_size = 1048576

        self.loading_checkpoint = False
        self.viewing_output = False

        self.total_frames = int(total_frames / num_envs)  # This needs to be divided by replay_period
        # this is the number of gradient steps! not number of frames

        if not self.loading_checkpoint:
            self.per_beta = 0.4

        # target_net, ema, trust_region
        self.stabiliser = "target_net"

        if self.stabiliser == "ema":
            self.soft_updates = True
        else:
            self.soft_updates = False

        # NOT IMPLEMENTED
        if self.stabiliser == "trust_regions":
            self.trust_regions = True
        else:
            self.trust_regions = False

        self.soft_update_tau = 0.001  # 0.001 for non-sample-eff
        self.replace_target_cnt = 8000  # This is the number of grad steps - could be a little jank
        # when changing num_envs/batch size/replay ratio

        # NOT IMPLEMENTED
        self.tr_alpha = 1
        self.tr_period = 1500

        self.loss_type = "huber"  # NOT IMPLEMENTED

        if self.iqn:
            self.num_tau = 8

        self.per_alpha = 0.2
        if self.loading_checkpoint:
            self.per_beta = 0.8
            self.min_sampling_size = 300000

        #c51
        self.Vmax = 10
        self.Vmin = -10
        self.N_ATOMS = 51

        if not self.noisy:
            if not self.loading_checkpoint and not self.testing:
                self.eps_start = 1.0
                self.eps_steps = (self.replay_ratio * 500000) / num_envs
                self.eps_final = 0.01
            else:
                self.eps_start = 0.01
                self.eps_steps = 250000
                self.eps_final = 0.01

            self.epsilon = EpsilonGreedy(self.eps_start, self.eps_steps, self.eps_final, self.action_space)

        self.num_envs = num_envs
        self.memories = []
        for i in range(num_envs):
            self.memories.append(ReplayMemory(self.max_mem_size // num_envs, self.n, self.gamma, device, alpha=self.per_alpha, beta=self.per_beta))

        if self.impala:
            if not self.iqn:
                self.net = ImpalaCNNLarge(self.input_dims[0], self.n_actions, atoms=self.N_ATOMS, Vmin=self.Vmin, Vmax=self.Vmax,
                                             device=self.device, noisy=self.noisy, spectral=self.spectral_norm, c51=self.c51,
                                          maxpool=self.maxpool, model_size=self.model_size)

                self.tgt_net = ImpalaCNNLarge(self.input_dims[0], self.n_actions, atoms=self.N_ATOMS, Vmin=self.Vmin, Vmax=self.Vmax,
                                                 device=self.device, noisy=self.noisy, spectral=self.spectral_norm,
                                              c51=self.c51, maxpool=self.maxpool, model_size=self.model_size)
            else:
                self.net = ImpalaCNNLargeIQN(self.input_dims[0], self.n_actions,spectral=self.spectral_norm, device=self.device,
                                             noisy=self.noisy, maxpool=self.maxpool, model_size=self.model_size, num_tau=self.num_tau, maxpool_size=self.maxpool_size)

                self.tgt_net = ImpalaCNNLargeIQN(self.input_dims[0], self.n_actions,spectral=self.spectral_norm, device=self.device,
                                             noisy=self.noisy, maxpool=self.maxpool, model_size=self.model_size, num_tau=self.num_tau, maxpool_size=self.maxpool_size)
        else:
            self.net = NatureIQN(self.input_dims[0], self.n_actions, device=self.device,
                                     noisy=self.noisy, num_tau=self.num_tau)

            self.tgt_net = NatureIQN(self.input_dims[0], self.n_actions, device=self.device,
                                     noisy=self.noisy, num_tau=self.num_tau)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, eps=0.005 / self.batch_size)  # 0.00015

        self.net.train()
        #self.tgt_net.train()

        for param in self.tgt_net.parameters():
            param.requires_grad = False

        self.env_steps = 0
        self.grad_steps = 0

        self.replay_ratio_cnt = 0
        self.eval_mode = False

        if self.loading_checkpoint:
            self.load_models()

        self.total_grad_steps = self.total_frames / (self.num_envs / self.replay_ratio)

        self.priority_weight_increase = (1 - self.per_beta) / self.total_grad_steps

    def get_grad_steps(self):
        return self.grad_steps

    def set_eval_mode(self):
        self.net.eval()
        self.tgt_net.eval()
        self.eval_mode = True

    def set_train_mode(self):
        self.net.train()
        self.tgt_net.train()
        self.eval_mode = False

    def choose_action(self, observation):
        with T.no_grad():
            if self.noisy:
                self.net.reset_noise()

            #state = T.tensor(np.array(list(observation)), dtype=T.float).to(self.net.device)
            state = T.tensor(observation, dtype=T.float).to(self.net.device)
            #state = state.cuda()
            qvals = self.net.qvals(state, advantages_only=True)
            x = T.argmax(qvals, dim=1).cpu()
            # this should contain (num_envs) different actions

            if not self.noisy and not self.eval_mode:
                for i in range(len(observation)):
                    action = self.epsilon.choose_action()
                    if action is not None:
                        x[i] = action

            if self.eval_mode:
                for i in range(len(observation)):
                    if np.random.random() > 0.99:
                        x[i] = np.random.choice(self.action_space)

            return x

    def store_transition(self, state, action, reward, done, stream, prio=None):
        if prio is None:

            self.memories[stream].append(torch.from_numpy(state), action, reward, done)
        else:
            self.memories[stream].append(torch.from_numpy(state), action, reward, done, 0)
        self.env_steps += 1

    def replace_target_network(self):
        self.tgt_net.load_state_dict(self.net.state_dict())

    def save_model(self):
        self.net.save_checkpoint(self.agent_name + str(self.env_steps))

    def load_models(self):
        self.net.load_checkpoint()
        self.tgt_net.load_checkpoint()


    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        with torch.no_grad():
            for target_param, local_param in zip(self.tgt_net.parameters(), self.net.parameters()):
                target_param.data.copy_(self.soft_update_tau*local_param.data + (1.0-self.soft_update_tau)*target_param.data)

    def learn(self):
        if self.replay_ratio < 1:
            if self.replay_ratio_cnt == 0:
                self.learn_call()
            self.replay_ratio_cnt = (self.replay_ratio_cnt + 1) % (int(1 / self.replay_ratio))
        else:
            for i in range(self.replay_ratio):
                self.learn_call()

    def learn_call(self):

        if self.env_steps < self.min_sampling_size:
            return

        for i in range(self.num_envs):
            self.memories[i].priority_weight = min(self.memories[i].priority_weight + self.priority_weight_increase, 1)

        self.optimizer.zero_grad()

        if not self.soft_updates:
            if self.grad_steps % self.replace_target_cnt == 0:
                self.replace_target_network()
        else:
            self.soft_update()

        try:

            # get total priority from each tree
            buffer_totals = np.empty(self.num_envs, dtype=float)
            for i in range(self.num_envs):
                buffer_totals[i] = self.memories[i].transitions.total()

            # create probability distribution based on prios
            buffer_dist = buffer_totals / buffer_totals.sum()

            if self.per_splits > 1:
                mems = np.random.choice(self.num_envs, self.per_splits, replace=False, p=buffer_dist)

                idxs, states, actions, rewards, next_states, dones, weights = self.memories[mems[0]].sample(
                    self.batch_size // self.per_splits)

                for i in range(self.per_splits - 1):

                    idxsN, statesN, actionsN, rewardsN, next_statesN, donesN, weightsN = self.memories[mems[i + 1]].sample(
                        self.batch_size // self.per_splits)

                    idxs = np.concatenate((idxs, idxsN))
                    states = torch.cat((states, statesN))
                    actions = torch.cat((actions, actionsN))
                    rewards = torch.cat((rewards, rewardsN))
                    next_states = torch.cat((next_states, next_statesN))
                    dones = torch.cat((dones, donesN))
                    weights = torch.cat((weights, weightsN))

            else:
                mem = np.random.choice(self.num_envs, self.per_splits, replace=False, p=buffer_dist)[0]
                idxs, states, actions, rewards, next_states, dones, weights = self.memories[mem].sample(
                    self.batch_size)

        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            print("Infinity Error?")
            raise Exception("stop")
            return

        states = states.clone().detach().to(self.net.device)
        rewards = rewards.clone().detach().to(self.net.device)
        dones = dones.clone().detach().to(self.net.device).squeeze()
        actions = actions.clone().detach().to(self.net.device)
        states_ = next_states.clone().detach().to(self.net.device)

        #use this code to check your states are correct
        """
        plt.imshow(states[0][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        plt.show()

        plt.imshow(states[1][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        plt.show()

        plt.imshow(states[2][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        plt.show()
        """

        if self.noisy:
            with torch.no_grad():
                self.tgt_net.reset_noise()

        if self.c51:
            distr_v, qvals_v = self.net.both(states)
            state_action_values = distr_v[range(self.batch_size), actions.data]
            state_log_sm_v = F.log_softmax(state_action_values, dim=1)

            with torch.no_grad():
                next_distr_v, next_qvals_v = self.tgt_net.both(states_)
                action_distr_v, action_qvals_v = self.net.both(states_)

                next_actions_v = action_qvals_v.max(1)[1]

                next_best_distr_v = next_distr_v[range(self.batch_size), next_actions_v.data]
                next_best_distr_v = self.tgt_net.apply_softmax(next_best_distr_v)
                next_best_distr = next_best_distr_v.data.cpu()

                proj_distr = distr_projection(next_best_distr, rewards.cpu(), dones.cpu(), self.Vmin, self.Vmax, self.N_ATOMS,
                                              self.gamma ** self.n)

                proj_distr_v = proj_distr.to(self.net.device)

            loss_v = -state_log_sm_v * proj_distr_v
            weights = T.squeeze(weights)
            loss_v = weights.to(self.net.device) * loss_v.sum(dim=1)

            loss = loss_v.mean()

        elif not self.iqn:
            indices = np.arange(self.batch_size)

            q_pred = self.net.forward(states)
            q_targets = self.tgt_net.forward(states_)
            q_actions = self.net.forward(states_)

            q_pred = q_pred[indices, actions]

            with torch.no_grad():
                max_actions = T.argmax(q_actions, dim=1)
                q_targets[dones] = 0.0

                q_target = rewards + (self.gamma ** self.n) * q_targets[indices, max_actions]

            td_error = q_target - q_pred
            loss_v = (td_error.pow(2) * weights.to(self.net.device))
            loss = loss_v.mean().to(self.net.device)

        elif self.iqn and not self.munchausen:

            Q_targets_next, _ = self.tgt_net(states_)

            if self.double: #this may be wrong - seems to perform better without. Could just be chance though
                indices = np.arange(self.batch_size)
                q_actions = self.net.qvals(states_)
                max_actions = T.argmax(q_actions, dim=1)
                Q_targets_next = Q_targets_next[indices,: ,max_actions].detach().unsqueeze(1)
            else:
                Q_targets_next = Q_targets_next.detach().max(2)[0].unsqueeze(1)  # (batch_size, 1, N)

            actions = actions.unsqueeze(1)
            rewards = rewards.unsqueeze(1)
            dones = dones.unsqueeze(1)
            weights = weights.unsqueeze(1)

            # Compute Q targets for current states
            Q_targets = rewards.unsqueeze(-1) + (
                        self.gamma ** self.n * Q_targets_next * (~dones.unsqueeze(-1)))

            # Get expected Q values from local model
            Q_expected, taus = self.net(states)
            Q_expected = Q_expected.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.num_tau, 1))

            # Quantile Huber loss
            td_error = Q_targets - Q_expected
            loss_v = torch.abs(td_error).sum(dim=1).mean(dim=1).data
            assert td_error.shape == (self.batch_size, self.num_tau, self.num_tau), "wrong td error shape"
            huber_l = calculate_huber_loss(td_error, 1.0)
            quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

            loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True)  # , keepdim=True if per weights get multipl
            loss = loss * weights.to(self.net.device)
            loss = loss.mean()

        elif self.iqn and self.munchausen:
            Q_targets_next, _ = self.tgt_net(next_states)
            Q_targets_next = Q_targets_next.detach()  # (batch, num_tau, actions)
            q_t_n = Q_targets_next.mean(dim=1)

            actions = actions.unsqueeze(1)
            rewards = rewards.unsqueeze(1)
            dones = dones.unsqueeze(1)
            weights = weights.unsqueeze(1)

            # calculate log-pi
            logsum = torch.logsumexp(
                (q_t_n - q_t_n.max(1)[0].unsqueeze(-1)) / self.entropy_tau, 1).unsqueeze(-1)  # logsum trick
            #assert logsum.shape == (self.batch_size, 1), "log pi next has wrong shape: {}".format(logsum.shape)
            tau_log_pi_next = (q_t_n - q_t_n.max(1)[0].unsqueeze(-1) - self.entropy_tau * logsum).unsqueeze(1)

            pi_target = F.softmax(q_t_n / self.entropy_tau, dim=1).unsqueeze(1)

            Q_target = (self.gamma ** self.n * (
                        pi_target * (Q_targets_next - tau_log_pi_next) * (~dones.unsqueeze(-1))).sum(2)).unsqueeze(1)
            #assert Q_target.shape == (self.batch_size, 1, self.num_tau)

            q_k_target = self.net.qvals(states).detach()
            v_k_target = q_k_target.max(1)[0].unsqueeze(-1)
            tau_log_pik = q_k_target - v_k_target - self.entropy_tau * torch.logsumexp(
                (q_k_target - v_k_target) / self.entropy_tau, 1).unsqueeze(-1)

            #assert tau_log_pik.shape == (self.batch_size, self.n_actions), "shape instead is {}".format(
                #tau_log_pik.shape)
            munchausen_addon = tau_log_pik.gather(1, actions)

            # calc munchausen reward:
            munchausen_reward = (rewards + self.alpha * torch.clamp(munchausen_addon, min=self.lo, max=0)).unsqueeze(-1)
            #assert munchausen_reward.shape == (self.batch_size, 1, 1)
            # Compute Q targets for current states
            Q_targets = munchausen_reward + Q_target
            # Get expected Q values from local model
            q_k, taus = self.net(states)
            Q_expected = q_k.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.num_tau, 1))
            #assert Q_expected.shape == (self.batch_size, self.num_tau, 1)

            # Quantile Huber loss
            td_error = Q_targets - Q_expected
            loss_v = torch.abs(td_error).sum(dim=1).mean(dim=1).data
            #assert td_error.shape == (self.batch_size, self.num_tau, self.num_tau), "wrong td error shape"
            huber_l = calculate_huber_loss(td_error, 1.0)
            quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

            loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True)  # , keepdim=True if per weights get multipl
            loss = loss * weights.to(self.net.device)
            loss = loss.mean()

        loss.backward()
        T.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.optimizer.step()

        if not self.noisy:
            self.epsilon.update_eps()

        self.grad_steps += 1
        if self.grad_steps % 10000 == 0:
            print("Completed " + str(self.grad_steps) + " gradient steps")

        if self.num_envs > 1 and self.per_splits > 1:
            idxs = np.split(idxs, self.per_splits)
            loss_v = torch.split(loss_v, self.batch_size // self.per_splits)
            for i in range(self.per_splits):
                self.memories[mems[i]].update_priorities(idxs[i], loss_v[i].cpu().detach().numpy())
        else:
            self.memories[mem].update_priorities(idxs, loss_v.cpu().detach().numpy())


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 8, 8), "huber loss has wrong shape"
    return loss

def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = T.zeros((batch_size, n_atoms), dtype=T.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).type(T.int64)
        u = np.ceil(b_j).type(T.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).type(T.int64)
        u = np.ceil(b_j).type(T.int64)
        eq_mask = u == l
        eq_dones = T.clone(dones)
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = T.clone(dones)
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr
