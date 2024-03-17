import torch
from static_networks import ImpalaCNNLargeIQN
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
# load models

model1 = ImpalaCNNLargeIQN(4, 18, spectral=True, device=torch.device("cuda:0"),
                                             noisy=False, maxpool=True, model_size=2, num_tau=8, maxpool_size=6,
                                             dueling=True, sqrt=False, ede=False, moe=False, pruning=False, linear_size=1024)

"""model2 = ImpalaCNNLargeIQN(4, 18, spectral=True, device=torch.device("cuda:0"),
                                             noisy=False, maxpool=True, model_size=2, num_tau=8, maxpool_size=6,
                                             dueling=True, sqrt=False, ede=False, moe=False, pruning=False, linear_size=512)"""

model1.load_checkpoint("BTR_1024.model")

def get_activation(name):
    global outputs
    def hook(model, input, output):
        outputs[name] = output.detach()

    return hook


for name, layer in model1.named_modules():
    if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
        layer.register_forward_hook(get_activation(name))


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

    return counts

def qval_var(tensor):
    return torch.mean(torch.var(tensor, dim=1)).item()

def action_var(tensor):
    counts = freq_per_action(tensor)

    return torch.var(counts.float()).item()


if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(os.getcwd()), '..\\states_dataset.pt')
    file_type = 'pt'  # or 'pt' for PyTorch tensor files

    dataset_from_file = StatesDatasetFromFile(file_path, file_type)

    batch_size = 512

    dataloader = DataLoader(dataset_from_file,
                            batch_size=batch_size,  # according to your device memory
                            shuffle=True)  # Don't forget to seed your dataloader

    device = torch.device("cuda:0")
    inputs = next(iter(dataloader)).squeeze().to(device)

    dormant_tau = 0.025
    outputs = {}

    # get number of dormants
    all_dormants = 0
    total_neurons = 0
    qvals = model1.qvals(inputs)
    for key, value in outputs.items():
        values = torch.reshape(value, (batch_size, -1))
        values = torch.mean(values, dim=0)

        values = torch.abs(values) / (torch.sum(torch.abs(values)) * (1 / len(values)))

        dormants = values <= dormant_tau

        dormant_total = torch.sum(dormants)

        all_dormants += dormant_total
        total_neurons += len(dormants)

    dormant_frac = all_dormants / total_neurons

    print(dormant_frac.item())

    # show distribution

    max_val = 0.8
    num_bins = 20

    inc = max_val / num_bins

    qvals = model1.qvals(inputs)
    for key, value in outputs.items():
        values = torch.reshape(value, (batch_size, -1))
        values = torch.mean(values, dim=0)

        values = torch.abs(values) / (torch.sum(torch.abs(values)) * (1 / len(values)))

        if 'bins' in locals():
            new_bins = torch.histc(values, bins=num_bins, min=0, max=max_val + .0001)
            bins += new_bins
        else:
            bins = new_bins = torch.histc(values, bins=num_bins, min=0, max=max_val + .0001)


    #for i in bins
    bins = bins.cpu()
    plt.figure(figsize=[12, 6])
    bin_labels = [f"[{round(i * inc, 3)}, {round((i + 1) * inc, 3)})" for i in range(num_bins - 1)] + [str(max_val) + '+']
    plt.bar(range(num_bins), bins, tick_label=bin_labels)
    plt.xlabel('Value ranges')
    plt.ylabel('Counts')
    plt.title('Histogram of Value Bins')
    plt.show()

