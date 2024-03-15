from torch_cka import CKA
import torch
from networks import ImpalaCNNLargeIQN

# load models

model1 = ImpalaCNNLargeIQN(4, 18, spectral=True, device=torch.device("cuda:0"),
                                             noisy=False, maxpool=True, model_size=2, num_tau=8, maxpool_size=6,
                                             dueling=True, sqrt=False, ede=False, moe=False, pruning=False, linear_size=1024)

model2 = ImpalaCNNLargeIQN(4, 18, spectral=True, device=torch.device("cuda:0"),
                                             noisy=False, maxpool=True, model_size=2, num_tau=8, maxpool_size=6,
                                             dueling=True, sqrt=False, ede=False, moe=False, pruning=False, linear_size=512)

model1.load_checkpoint("BTR_C500.model")
print("Hey\n\n\n")
model2.load_checkpoint("BTR_ema.model")
"""
dataloader = DataLoader(your_dataset,
                        batch_size=128, # according to your device memory
                        shuffle=False)  # Don't forget to seed your dataloader

cka = CKA(model1, model2,
          model1_name="C500",   # good idea to provide names to avoid confusion
          model2_name="EMA",
          device='cuda')

cka.compare(dataloader) # secondary dataloader is optional

results = cka.export()  # returns a dict that contains model names, layer names
                        # and the CKA matrix"""