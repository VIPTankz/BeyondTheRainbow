import torch
from static_networks import ImpalaCNNLargeIQN
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
# load models

model1 = ImpalaCNNLargeIQN(4, 18, spectral=True, device=torch.device("cuda:0"),
                                             noisy=False, maxpool=True, model_size=2, num_tau=8, maxpool_size=6,
                                             dueling=True, sqrt=False, ede=False, moe=False, pruning=False, linear_size=1024)

model2 = ImpalaCNNLargeIQN(4, 18, spectral=True, device=torch.device("cuda:0"),
                                             noisy=False, maxpool=True, model_size=2, num_tau=8, maxpool_size=6,
                                             dueling=True, sqrt=False, ede=False, moe=False, pruning=False, linear_size=512)

model1.load_checkpoint("BTR_1024.model")
model2.load_checkpoint("BTR_C500.model")

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

    dataloader = DataLoader(dataset_from_file,
                            batch_size=512,  # according to your device memory
                            shuffle=False)  # Don't forget to seed your dataloader


    device = torch.device("cuda:0")
    inputs = next(iter(dataloader)).squeeze().to(device)

    models = [model1, model2]

    outputs = []
    for model in models:
        outputs.append(model.qvals(inputs))

    print("Frequency of each action")
    for i in outputs:
        freq = freq_per_action(i)
        print(freq)

    print("\n")
    print("Q-Value Discrimination (Q-value Variance)")

    for i in outputs:
        var = qval_var(i)
        print(var)

    print("\n")
    print("Action Discrimination (Action Variance)")

    for i in outputs:
        var = action_var(i)
        print(var)
