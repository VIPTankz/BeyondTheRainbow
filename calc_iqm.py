import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd

human_scores = np.array([37188, -16, 8049, 7243, 13455], dtype=np.float64)

random_scores = np.array([2360, -19, 2292, 761, 164], dtype=np.float64)

# rainbow
rainbow_scores = np.array([40895, 22, 9229, 8605, 18503], dtype=np.float64)

iqn_scores = np.array([41475, 21, 9129, 5137, 16995], dtype=np.float64)

scores = np.array([52517, 22, 12761, 5327, 14739], dtype=np.float64)

# Create a NumPy array with a shape of "runs" x "game"

results = (scores - random_scores) / (human_scores - random_scores)
results = np.array([results])

print("IQM:")
print(round(scipy.stats.trim_mean(results, 0.25, axis=None),3))

print("Median:")
print(round(np.median(np.mean(results, axis=-2, keepdims=False), axis=-1), 3))

print("Mean:")
print(round(np.mean(np.mean(results, axis=-2, keepdims=False), axis=-1), 3))