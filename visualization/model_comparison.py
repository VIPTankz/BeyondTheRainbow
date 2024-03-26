import torch

import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Now you can import your module as if it is in the current directory
from networks import ImpalaCNNLargeIQN
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import matplotlib.pyplot as plt
# load models
from torchsummary import summary

"""for name, _ in model1.named_modules():
    print(name)
raise Exception("stop")"""

class StatesDatasetFromFile(Dataset):
    """Dataset for states loaded from a file."""

    def __init__(self, file_path, file_type='npy'):
        """
        Args:
            file_path (str): Path to the file containing the states.
            file_type (str): Type of the file ('npy' for numpy array or 'pt' for PyTorch tensor).
        """
        if file_type == 'npy':
            self.states = np.load(file_path)
        elif file_type == 'pt':
            self.states = torch.load(file_path)
        else:
            raise ValueError("Unsupported file type. Use 'npy' for numpy arrays or 'pt' for PyTorch tensors.")

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.states)

    def __getitem__(self, idx):
        """Generates one sample of data."""
        state = self.states[idx]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        elif not isinstance(state, torch.Tensor):
            raise TypeError("The dataset should contain numpy arrays or PyTorch tensors.")
        return state


def freq_per_action(tensor):
    # Find the indices of the maximum values for each entry along dimension 1
    argmax_indices = torch.argmax(tensor, dim=1)

    # Count the occurrences of each index
    counts = torch.bincount(argmax_indices, minlength=tensor.size(1))

    return counts.cpu().numpy()

def qval_var(tensor):
    return torch.mean(torch.var(tensor, dim=1)).item()

def action_var(tensor):
    counts = freq_per_action(tensor)

    return torch.var(counts.float()).item()


if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(os.getcwd()), '..\\states_dataset.pt')
    file_type = 'pt'  # or 'pt' for PyTorch tensor files

    dataset_from_file = StatesDatasetFromFile(file_path, file_type)

    dataloader = DataLoader(dataset_from_file,
                            batch_size=1024,  # according to your device memory
                            shuffle=False)  # Don't forget to seed your dataloader


    device = torch.device("cuda:0")
    inputs = next(iter(dataloader)).squeeze().to(device)

    model = ImpalaCNNLargeIQN(4, 18, spectral=True, device=torch.device("cuda:0"),
                              noisy=False, maxpool=True, model_size=2, num_tau=8, maxpool_size=6,
                              dueling=True, sqrt=False, ede=False, moe=False, pruning=False, linear_size=512,
                              spectral_lin=True)
    summary(model, (4, 84, 84))
    # Number of models
    num_models = 24

    # This will store the frequency arrays from all models
    all_freqs = []
    all_qvals = []

    # Load each model, compute q-values, then frequencies
    for i in range(1, num_models + 1):  # Start from 1 to 18
        model.load_checkpoint(f"models\\BTR_BattleZone40M_lin_size1024_noisy0_spec_lin1_munch1_double0_{i}M.model")
        qvals = model.qvals(inputs)
        all_qvals.append(torch.mean(qvals, dim=0).detach().cpu().numpy())
        freq = freq_per_action(qvals)
        all_freqs.append(freq)
        print("Finished Model " + str(i))

    all_freqs2 = []
    all_qvals2 = []

    model = ImpalaCNNLargeIQN(4, 18, spectral=True, device=torch.device("cuda:0"),
                              noisy=False, maxpool=True, model_size=2, num_tau=8, maxpool_size=6,
                              dueling=True, sqrt=False, ede=False, moe=False, pruning=False, linear_size=1024,
                              spectral_lin=False)



    # Load each model, compute q-values, then frequencies
    for i in range(1, num_models + 1):  # Start from 1 to 18
        model.load_checkpoint(f"models2\\BTR_BattleZone80M_lin_size1024_noisy0_spec_lin0_munch1_double0_{i}M.model")
        qvals = model.qvals(inputs)
        all_qvals2.append(torch.mean(qvals, dim=0).detach().cpu().numpy())
        freq = freq_per_action(qvals)
        all_freqs2.append(freq)
        print("Finished Model " + str(i))

    # Set up the figure for animation
    #bins = np.arange(len(all_freqs[0])) - 0.5  # Assuming all_freqs elements are integers

    def animate(num, data, data2):
        plt.cla()

        # Number of groups
        n_groups = len(data[0])

        # Create bar locations for the two datasets
        # Setting the bar width
        bar_width = 0.35

        # Creating the index for the groups
        index = np.arange(n_groups)

        # Plotting the first set of bars
        plt.bar(index, data[num], bar_width, label=labels[0])

        # Plotting the second set of bars, offset by the width of a bar
        plt.bar(index + bar_width, data2[num], bar_width, label=labels[1])

        # Additional options
        plt.xlabel('Action')
        plt.ylabel('Times used')
        plt.xticks(index + bar_width / 2, [i for i in
                                           range(n_groups)])  # Setting x-ticks to be in the middle of the grouped bars
        plt.legend()

        # Make sure the x-axis accommodates all bars
        plt.xlim(-0.5, n_groups)
        plt.title(str(num) + " Million Frames")


    """    data = all_freqs
    data2 = all_freqs2

    fig = plt.figure()
    # Create the animation
    ani = FuncAnimation(fig, animate, frames=num_models, repeat=False, fargs=(data, data2))
    writergif = animation.PillowWriter(fps=1)

    # Save the animation
    ani.save('model_frequencies.gif', writer=writergif)  # Adjust fps for speed of animation

    plt.show()"""

    labels = ["Model1", "Model2"]

    data = all_qvals
    data2 = all_qvals2

    # this part does Q-vals per action
    fig = plt.figure()
    # Create the animation
    ani = FuncAnimation(fig, animate, frames=num_models, repeat=False, fargs=(data, data2))
    writergif = animation.PillowWriter(fps=1)

    # Save the animation
    ani.save('model_qvals.gif', writer=writergif)  # Adjust fps for speed of animation

    plt.show()

    """print("\n")
    print("Q-Value Discrimination (Q-value Variance)")

    var = qval_var(qvals)
    print(var)


    print("\n")
    print("Action Discrimination (Action Variance)")

    var = action_var(qvals)
    print(var)"""
