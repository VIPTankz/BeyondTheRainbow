
import os
import numpy as np
from static_networks import ImpalaCNNLargeIQN
import torch
from torchsummary import summary
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Fully connected layer is not needed
    model1 = ImpalaCNNLargeIQN(4, 18, spectral=True, device=torch.device("cuda:0"),
                               noisy=False, maxpool=True, model_size=2, num_tau=8, maxpool_size=6,
                               dueling=True, sqrt=False, ede=False, moe=False, pruning=False, linear_size=1024)

    """model2 = ImpalaCNNLargeIQN(4, 18, spectral=True, device=torch.device("cpu"),
                               noisy=False, maxpool=True, model_size=2, num_tau=8, maxpool_size=6,
                               dueling=True, sqrt=False, ede=False, moe=False, pruning=False, linear_size=512)"""

    model1.load_checkpoint("BTR_1024.model")
    #model2.load_checkpoint("BTR_C2000.model")
    summary(model1, (4, 84, 84))

    delta = 0.01

    layers = ["dueling.value_branch.0", "dueling.advantage_branch.0"]


    for name, module in model1.named_modules():

        if hasattr(module, "weight") and not name.endswith("parametrizations") and name in layers:
            print(name)
            matrix = module.weight.detach().cpu().numpy()

            # singular values - np.linalg.svd
            U, S, V = np.linalg.svd(matrix)
            print(S.shape)

            cumsum = np.cumsum(S) / np.sum(S)

            plt.plot(np.arange(len(S)), cumsum, label=name)

            values = np.argmax(cumsum > 1 - delta)
            print(f"{values=}")
            print("Totals: " + str(len(cumsum)))
            print()

    plt.axhline(0.99)
    plt.legend()
    plt.show()


    # plt.plot(np.arange(features), np.cumsum / np.sum, label="""
    """if name.endswith("weight"):
        print(name)
        matrix = module.weight.detach().cpu().numpy()

        # singular values - np.linalg.svd
        U, S, V = np.linalg.svd(matrix)

        cumsum = np.cumsum(S) / np.sum(S)

        values = np.argmax(cumsum > 1 - delta)
        print(f"{values=}")

        # plt.plot(np.arange(features), np.cumsum / np.sum, label="""

