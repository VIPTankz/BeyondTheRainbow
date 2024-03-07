"""
This file defines all the neural network architectures available to use.
"""
from functools import partial
from math import sqrt
import math

import torch
from torch import nn as nn, Tensor
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import time
import torch.nn.utils.prune as prune
#from soft_moe import SoftMoELayerWrapper

#import kornia
from torchvision.utils import save_image

class FactorizedNoisyLinear(nn.Module):
    """ The factorized Gaussian noise layer for noisy-nets dqn. """
    def __init__(self, in_features: int, out_features: int, sigma_0: float) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        # weight: w = \mu^w + \sigma^w . \epsilon^w
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        # bias: b = \mu^b + \sigma^b . \epsilon^b
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # initialization is similar to Kaiming uniform (He. initialization) with fan_mode=fan_in
        scale = 1 / sqrt(self.in_features)

        init.uniform_(self.weight_mu, -scale, scale)
        init.uniform_(self.bias_mu, -scale, scale)

        init.constant_(self.weight_sigma, self.sigma_0 * scale)
        init.constant_(self.bias_sigma, self.sigma_0 * scale)

    @torch.no_grad()
    def _get_noise(self, size: int) -> Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        # f(x) = sgn(x)sqrt(|x|)
        return noise.sign().mul_(noise.abs().sqrt_())

    @torch.no_grad()
    def reset_noise(self) -> None:
        # like in eq 10 and 11 of the paper
        epsilon_in = self._get_noise(self.in_features)
        epsilon_out = self._get_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @torch.no_grad()
    def disable_noise(self) -> None:
        self.weight_epsilon[:] = 0
        self.bias_epsilon[:] = 0

    def forward(self, input: Tensor) -> Tensor:
        # y = wx + d, where
        # w = \mu^w + \sigma^w * \epsilon^w
        # b = \mu^b + \sigma^b * \epsilon^b
        return F.linear(input,
                        self.weight_mu + self.weight_sigma*self.weight_epsilon,
                        self.bias_mu + self.bias_sigma*self.bias_epsilon)

class NatureIQN(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, in_depth, actions, device='cuda:0',
                 noisy=False, num_tau=8):
        super().__init__()

        self.start = time.time()
        self.actions = actions
        self.device = device
        self.noisy = noisy

        self.linear_size = 256
        self.num_tau = num_tau

        self.n_cos = 64
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(device)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_depth, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.conv_out_size = 3136

        self.cos_embedding = nn.Linear(self.n_cos, self.conv_out_size)

        if not self.noisy:
            self.fc1 = nn.Linear(self.conv_out_size, 256)
            self.fc2 = nn.Linear(256, self.actions)

        else:
            self.fc1 = NoisyLinear(self.conv_out_size, 256)
            self.fc2 = NoisyLinear(256, self.actions)

        self.to(device)

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, input):
        """
        Quantile Calculation depending on the number of tau

        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]

        """
        input = input.float() / 256
        batch_size = input.size()[0]

        x = self.conv(input)
        x = x.view(batch_size, -1)

        cos, taus = self.calc_cos(batch_size, self.num_tau)  # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size * self.num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, self.num_tau, self.conv_out_size)  # (batch, n_tau, layer)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * self.num_tau, self.conv_out_size)

        x = torch.relu(self.fc1(x))
        out = self.fc2(x)

        return out.view(batch_size, self.num_tau, self.actions), taus

    def qvals(self, inputs, advantages_only=None):
        quantiles, _ = self.forward(inputs)
        actions = quantiles.mean(dim=1)
        return actions

    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).to(self.device).unsqueeze(-1) #(batch_size, n_tau, 1)
        cos = torch.cos(taus*self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos, taus

    def save_checkpoint(self, name):
        #print('... saving checkpoint ...')
        torch.save(self.state_dict(), name)

    def load_checkpoint(self):
        #print('... loading checkpoint ...')
        print("Loaded Checkpoint!")
        self.load_state_dict(torch.load("current_model179609"))

class Dueling(nn.Module):
    """ The dueling branch used in all nets that use dueling-dqn. """
    def __init__(self, value_branch, advantage_branch):
        super().__init__()
        self.flatten = nn.Flatten()
        self.value_branch = value_branch
        self.advantage_branch = advantage_branch

    #@torch.autocast('cuda')
    def forward(self, x, advantages_only=False):
        x = self.flatten(x)
        advantages = self.advantage_branch(x)
        if advantages_only:
            return advantages

        value = self.value_branch(x)
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))

    def reset_noise(self):
        self.value_branch[0].reset_noise()
        self.value_branch[2].reset_noise()
        self.advantage_branch[0].reset_noise()
        self.advantage_branch[2].reset_noise()

class DuelingAlt(nn.Module):
    """ The dueling branch used in all nets that use dueling-dqn. """
    def __init__(self, l1, l2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            l1,
            nn.ReLU(),
            l2
        )

    def forward(self, x, advantages_only=False):
        res = self.main(x)
        advantages = res[:, 1:]
        value = res[:, 0:1]
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))

class NatureCNN(nn.Module):
    """
    This is the CNN that was introduced in Mnih et al. (2013) and then used in a lot of later work such as
    Mnih et al. (2015) and the Rainbow paper. This implementation only works with a frame resolution of 84x84.
    """
    def __init__(self, depth, actions, linear_layer):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            linear_layer(3136, 512),
            nn.ReLU(),
            linear_layer(512, actions),
        )

    def forward(self, x, advantages_only=None):
        return self.main(x)


class DuelingNatureCNN(nn.Module):
    """
    Implementation of the dueling architecture introduced in Wang et al. (2015).
    This implementation only works with a frame resolution of 84x84.
    """
    def __init__(self, depth, actions, linear_layer):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.dueling = Dueling(
                nn.Sequential(linear_layer(3136, 512),
                              nn.ReLU(),
                              linear_layer(512, 1)),
                nn.Sequential(linear_layer(3136, 512),
                              nn.ReLU(),
                              linear_layer(512, actions))
            )

    def forward(self, x, advantages_only=False):
        f = self.main(x)
        return self.dueling(f, advantages_only=advantages_only)


class ImpalaCNNSmall(nn.Module):
    """
    Implementation of the small variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, depth, actions):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((6, 6))

        self.dueling = Dueling(
                nn.Sequential(dqn_model.NoisyLinear(1152, 256),
                              nn.ReLU(),
                              dqn_model.NoisyLinear(256, 1)),
                nn.Sequential(dqn_model.NoisyLinear(1152, 256),
                              nn.ReLU(),
                              dqn_model.NoisyLinear(256, actions))
            )

    def _get_conv_out(self, shape):
        o = self.main(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, advantages_only=False):
        x = x.float() / 256
        f = self.main(x)
        f = self.pool(f)
        return self.dueling(f, advantages_only=advantages_only)


class ImpalaCNNResidual(nn.Module):
    """
    Simple residual block used in the large IMPALA CNN.
    """
    def __init__(self, depth, norm_func):
        super().__init__()

        self.relu = nn.ReLU()

        self.conv_0 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))
        self.conv_1 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))

    #@torch.autocast('cuda')
    def forward(self, x):
        #if x.abs().sum().item() != 0:
        x_ = self.conv_0(self.relu(x))
        #if x_.abs().sum().item() == 0:
        #raise Exception("0 tensor found within residual layer!")
        x_ = self.conv_1(self.relu(x_))
        return x + x_


class ImpalaCNNBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    """
    def __init__(self, depth_in, depth_out, norm_func):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_0 = ImpalaCNNResidual(depth_out, norm_func=norm_func)
        self.residual_1 = ImpalaCNNResidual(depth_out, norm_func=norm_func)

    #@torch.autocast('cuda')
    def forward(self, x):
        x = self.conv(x)
        #if x.abs().sum().item() == 0:
        #raise Exception("Tensor output all zeros")
        #This bug still exists -- this sometimes outputs all 0s
        # Bug is now FIXED (I HOPE, it still lives in my nightmares)
        #turned out it in choose action? had to do action.cpu

        #raise Exception("Array of 0s!")
        #print(x.abs().sum().item())
        x = self.max_pool(x)

        x = self.residual_0(x)

        x = self.residual_1(x)

        return x


"""
self.input_dims[0], self.n_actions, atoms=self.N_ATOMS, Vmin=self.Vmin, Vmax=self.Vmax,
                                             device=self.device, noisy=self.noisy, spectral=self.spectral_norm, c51=self.c51,
                                          maxpool=self.maxpool, model_size=self.model_size
                                          
TODO
Need to move these arguments into this model
"""

class ImpalaCNNLarge(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    No IQN or C51
    """
    def __init__(self, in_depth, actions, model_size=2, spectral=True, noisy=False, maxpool=True, maxpool_size=6, device='cuda:0'):
        super().__init__()

        self.start = time.time()
        self.model_size = model_size
        self.actions = actions
        self.maxpool = maxpool
        self.maxpool_size = maxpool_size
        self.device = device

        if noisy:
            linear_layer = NoisyLinear
        else:
            linear_layer = nn.Linear

        def identity(p): return p

        norm_func = torch.nn.utils.spectral_norm if spectral else identity

        self.main = nn.Sequential(
            ImpalaCNNBlock(in_depth, 16*model_size, norm_func=norm_func),
            ImpalaCNNBlock(16*model_size, 32*model_size, norm_func=norm_func),
            ImpalaCNNBlock(32*model_size, 32*model_size, norm_func=norm_func),
            nn.ReLU()
        )

        if self.maxpool:
            self.pool = torch.nn.AdaptiveMaxPool2d((self.maxpool_size, self.maxpool_size))
            if self.maxpool_size == 8:
                self.conv_out_size = 2048*model_size
            elif self.maxpool_size == 6:
                self.conv_out_size = 1152*model_size
            elif self.maxpool_size == 4:
                self.conv_out_size = 512*model_size
            else:
                raise Exception("No Conv out size for this maxpool size")
        else:
            self.conv_out_size = 11520

        self.dueling = Dueling(
            nn.Sequential(linear_layer(self.conv_out_size, 256),
                          nn.ReLU(),
                          linear_layer(256, 1)),
            nn.Sequential(linear_layer(self.conv_out_size, 256),
                          nn.ReLU(),
                          linear_layer(256, actions))
        )

        self.to(device)

    def _get_conv_out(self, shape):
        o = self.main(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def reset_noise(self):
        self.dueling.reset_noise()

    def qvals(self, x, advantages_only=False):
        return self.forward(x, advantages_only)

    def forward(self, x, advantages_only=False):
        x = x.float() / 256
        """if test:
            save_image(x[0], 'img1.png')
            save_image(x[1], 'img2.png')
            save_image(x[2], 'img3.png')

            raise Exception("stop")"""

        f = self.main(x)
        if self.maxpool:
            f = self.pool(f)

        return self.dueling(f, advantages_only=advantages_only)

    def save_checkpoint(self):
        #print('... saving checkpoint ...')
        torch.save(self.state_dict(), "current_model" + str(int(time.time() - self.start)))

    def load_checkpoint(self):
        #print('... loading checkpoint ...')
        self.load_state_dict(torch.load("current_model420906"))

class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)

class ImpalaCNNLargeC51(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    No IQN
    """
    def __init__(self, in_depth, actions, model_size=2, spectral=True, atoms=51, Vmin=-10, Vmax=10, device='cuda:0',
                 noisy=False, maxpool=False):
        super().__init__()

        self.start = time.time()
        self.model_size = model_size
        self.actions = actions
        self.atoms = atoms
        self.device = device
        self.noisy = noisy
        self.c51 = c51
        self.iqn = iqn
        self.maxpool = maxpool
        if not self.c51:
            self.atoms = 1

        if spectral:
            spectral_norm = 'all'
        else:
            spectral_norm = 'none'

        DELTA_Z = (Vmax - Vmin) / (atoms - 1)

        def identity(p): return p

        norm_func = torch.nn.utils.spectral_norm if (spectral_norm == 'all') else identity
        norm_func_last = torch.nn.utils.spectral_norm if (spectral_norm == 'last' or spectral_norm == 'all') else identity

        self.conv = nn.Sequential(
            ImpalaCNNBlock(in_depth, 16*model_size, norm_func=norm_func),
            ImpalaCNNBlock(16*model_size, 32*model_size, norm_func=norm_func),
            ImpalaCNNBlock(32*model_size, 32*model_size, norm_func=norm_func_last),
            nn.ReLU()
        )

        if self.maxpool:
            self.pool = torch.nn.AdaptiveMaxPool2d((8, 8))
            conv_out_size = 2048*model_size
        else:
            conv_out_size = 11520

        if not self.noisy:
            self.fc1V = nn.Linear(conv_out_size, 256)
            self.fc1A = nn.Linear(conv_out_size, 256)
            self.fcV2 = nn.Linear(256, self.atoms)
            self.fcA2 = nn.Linear(256, actions * self.atoms)
        else:
            self.fc1V = NoisyLinear(conv_out_size, 256)
            self.fc1A = NoisyLinear(conv_out_size, 256)
            self.fcV2 = NoisyLinear(256, self.atoms)
            self.fcA2 = NoisyLinear(256, actions * self.atoms)

        if self.c51:
            self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
            self.softmax = nn.Softmax(dim=1)

        self.to(device)

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def fc_val(self, x):
        x = F.relu(self.fc1V(x))
        x = self.fcV2(x)

        return x

    def fc_adv(self, x):
        x = F.relu(self.fc1A(x))
        x = self.fcA2(x)

        return x

    def forward(self, x):
        if self.c51:
            batch_size = x.size()[0]
            fx = x.float() / 256
            conv_out = self.conv(fx)
            if self.maxpool:
                conv_out = self.pool(conv_out)

            conv_out = conv_out.view(batch_size, -1)

            val_out = self.fc_val(conv_out).view(batch_size, 1, self.atoms)
            adv_out = self.fc_adv(conv_out).view(batch_size, -1, self.atoms)
            adv_mean = adv_out.mean(dim=1, keepdim=True)
            return val_out + (adv_out - adv_mean)
        else:
            x = x.float() / 256
            batch_size = x.size()[0]

            f = self.conv(x)
            if self.maxpool:
                f = self.pool(f)

            f = f.view(batch_size, -1)

            V = self.fc_val(f)
            A = self.fc_adv(f)
            Q = V + A - A.mean(dim=1, keepdim=True)
            return Q

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        if self.c51:
            return self.both(x)[1]
        else:
            return self(x)

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, self.atoms)).view(t.size())

    def save_checkpoint(self):
        #print('... saving checkpoint ...')
        torch.save(self.state_dict(), "current_model3Ruins" + str(int(time.time() - self.start)))

    def load_checkpoint(self):
        #print('... loading checkpoint ...')
        self.load_state_dict(torch.load("finalAllTracks"))


class ImpalaCNNLargeIQN(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, in_depth, actions, model_size=2, spectral=True, device='cuda:0',
                 noisy=False, maxpool=False, num_tau=8, maxpool_size=6, dueling=True, sqrt=False, ede=False, moe=False,
                 pruning=False):
        super().__init__()

        self.start = time.time()
        self.model_size = model_size
        self.actions = actions
        self.device = device
        self.noisy = noisy
        self.maxpool = maxpool
        self.dueling = dueling
        self.sqrt = sqrt
        self.ede = ede
        self.moe = moe
        self.pruning = pruning

        if self.pruning:
            self.last_sparsity = 0

        if self.ede:
            self.ede_num_layers = 5

        if self.moe:
            self.output_channels = 32 * model_size

        self.linear_size = 512
        self.num_tau = num_tau

        self.maxpool_size = maxpool_size

        self.n_cos = 64
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(device)

        if noisy:
            linear_layer = NoisyLinear
        else:
            linear_layer = nn.Linear

        def identity(p): return p

        if spectral:
            norm_func = torch.nn.utils.parametrizations.spectral_norm
        else:
            norm_func = identity

        self.conv = nn.Sequential(
            ImpalaCNNBlock(in_depth, 16*model_size, norm_func=norm_func),
            ImpalaCNNBlock(16*model_size, 32*model_size, norm_func=norm_func),
            ImpalaCNNBlock(32*model_size, 32*model_size, norm_func=norm_func),
            nn.ReLU()
        )

        if self.maxpool:
            self.pool = torch.nn.AdaptiveMaxPool2d((self.maxpool_size, self.maxpool_size))
            if self.maxpool_size == 8:
                self.conv_out_size = 2048*model_size
            elif self.maxpool_size == 6:
                self.conv_out_size = 1152*model_size
            elif self.maxpool_size == 4:
                self.conv_out_size = 512*model_size
            else:
                raise Exception("No Conv out size for this maxpool size")
        else:
            self.conv_out_size = 11520

        self.cos_embedding = nn.Linear(self.n_cos, self.conv_out_size)

        if self.ede:
            self.ede_layers = []
            for i in range(self.ede_num_layers):
                self.ede_layers.append(Dueling(
                    nn.Sequential(linear_layer(self.conv_out_size, self.linear_size),
                                  nn.ReLU(),
                                  linear_layer(self.linear_size, 1)),
                    nn.Sequential(linear_layer(self.conv_out_size, self.linear_size),
                                  nn.ReLU(),
                                  linear_layer(self.linear_size, actions))
                ))

            for i in self.ede_layers:
                i.to(device)

        elif self.moe:

            """self.linear_layers = nn.Sequential(
                SoftMoELayerWrapper(
                    dim=self.conv_out_size // self.output_channels,
                    slots_per_expert=1,
                    num_experts=4,
                    layer=nn.Linear,
                    # nn.Linear arguments
                    in_features=self.conv_out_size // self.output_channels,
                    out_features=self.linear_size,
                ),
                nn.ReLU(),
                nn.Linear(self.linear_size, actions)
            )"""
            self.l1 = SoftMoELayerWrapper(
                    dim=self.conv_out_size // self.output_channels,
                    slots_per_expert=1,
                    num_experts=4,
                    layer=nn.Linear,
                    # nn.Linear arguments
                    in_features=self.conv_out_size // self.output_channels,
                    out_features=self.linear_size,
                )
            self.l2 = nn.ReLU()
            self.l3 = nn.Linear(self.linear_size, actions)

        elif self.dueling:
            self.dueling = Dueling(
                nn.Sequential(linear_layer(self.conv_out_size, self.linear_size),
                              nn.ReLU(),
                              linear_layer(self.linear_size, 1)),
                nn.Sequential(linear_layer(self.conv_out_size, self.linear_size),
                              nn.ReLU(),
                              linear_layer(self.linear_size, actions))
            )
        else:
            self.linear_layers = nn.Sequential(
                linear_layer(self.conv_out_size, self.linear_size),
                nn.ReLU(),
                linear_layer(self.linear_size, actions)
            )

        if self.pruning:
            self.parameters_to_prune = []
            for name, module in self.named_modules():
                if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
                    self.parameters_to_prune.append((module, 'weight'))
                if hasattr(module, 'bias') and module.bias is not None:
                    self.parameters_to_prune.append((module, 'bias'))

            self.parameters_to_prune = tuple(self.parameters_to_prune)

            """ unfinished erk scale stuff
            neurons_layer = 0
            neurons_prev_layer = 0
            kernel_w = 0
            kernel_h = 0

            # non-conv layer
            er_scaling =

            erk_scaling = 1 - (neurons_prev_layer + neurons_layer + kernel_w + kernel_h) / \
                        (neurons_prev_layer * neurons_layer * kernel_w * kernel_h)
            """

        self.to(device)

    def reset_noise(self):
        self.dueling.reset_noise()

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def get_prune_params_per_layer(self):
        pass

    def prune(self, sparsity):
        # loop over all modules in model

        # Pytorch pruning is based on proportion of UNPRUNED parameters, so doing 0.9 twice would result in
        # 99% of parameters being pruned
        self.last_sparsity = sparsity - self.last_sparsity

        # i[0] = module
        # i[1] = 'weight' or 'bias'
        # i[2] = amount (integer, parameters in that layer)
        for i in self.parameters_to_prune:

            prune.l1_unstructured(i[0], i[1], self.last_sparsity * i[3])


    #@torch.autocast('cuda')
    def forward(self, inputt, advantages_only=False):
        """
        Quantile Calculation depending on the number of tau

        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]

        """
        #print("Forward Func")
        inputt = inputt.float() / 255
        #print(input.abs().sum().item())
        batch_size = inputt.size()[0]

        x = self.conv(inputt)
        #print(x.device)
        if self.maxpool:
            x = self.pool(x)

        #print(x.device)

        x = x.view(batch_size, -1)

        cos, taus = self.calc_cos(batch_size, self.num_tau)  # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size * self.num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, self.num_tau, self.conv_out_size)  # (batch, n_tau, layer)

        if not self.moe:
            # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
            x = (x.unsqueeze(1) * cos_x).view(batch_size * self.num_tau, self.conv_out_size)

        if self.ede:
            outs = []
            for i in range(self.ede_num_layers):
                outs.append(self.ede_layers[i](x, advantages_only=advantages_only).view(batch_size, self.num_tau, self.actions))

            self.quantiles = torch.stack(outs, dim=0)  # should have shape (heads, bs, taus, actions)
            out = self.quantiles.mean(dim=0)
            #self.quantiles.detach()
        elif self.moe:
            # reshape x to be (batch_size * num_taus, channels, W*H)
            x = (x.unsqueeze(1) * cos_x).view(batch_size * self.num_tau, self.output_channels, self.conv_out_size // self.output_channels)
            print(x.shape)
            #out = self.linear_layers(x)

            out = self.l1(x)
            print(out.shape)
            out = self.l2(out)
            out = self.l3(out)
            print(out.shape)

        elif self.dueling:
            out = self.dueling(x, advantages_only=advantages_only)
        else:
            out = self.linear_layers(x)

        if self.sqrt:
            # square the Q-values but keep the sign
            squared_tensor = torch.square(out)
            out = squared_tensor * torch.sign(out)

        #print(out.device)

        if not self.ede:
            return out.view(batch_size, self.num_tau, self.actions), taus
        else:
            return out, taus

    #@torch.autocast('cuda')
    def qvals(self, inputs, advantages_only=False):
        quantiles, _ = self.forward(inputs, advantages_only)

        actions = quantiles.mean(dim=1)

        return actions

    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).to(self.device).unsqueeze(-1) #(batch_size, n_tau, 1)
        cos = torch.cos(taus*self.pis)

        #assert cos.shape == (batch_size, n_tau, self.n_cos), "cos shape is incorrect"
        return cos, taus

    def get_bootstrapped_uncertainty(self):
        # needs to do minibatch from multiple heads [M]
        # self.quantiles should be [heads, num_envs, taus, actions]
        eps_var = torch.permute(self.quantiles, (1, 0, 2, 3))  # [num_envs, heads, taus, actions]
        eps_var = torch.var(eps_var, dim=1)  # [num_envs, tau, action]
        eps_var = torch.mean(eps_var, dim=1)  # [B, action]
        return eps_var

    def save_checkpoint(self, name):
        #print('... saving checkpoint ...')
        torch.save(self.state_dict(), name + ".model")

    def load_checkpoint(self):
        #print('... loading checkpoint ...')
        self.load_state_dict(torch.load("current_model152582"))

def get_model(model_str, spectral_norm):
    if model_str == 'nature': return NatureCNN
    elif model_str == 'dueling': return DuelingNatureCNN
    elif model_str == 'impala_small': return ImpalaCNNSmall
    elif model_str.startswith('impala_large:'):
        return partial(ImpalaCNNLarge, model_size=int(model_str[13:]), spectral_norm=spectral_norm)