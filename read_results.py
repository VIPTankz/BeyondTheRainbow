import numpy as np
import matplotlib.pyplot as plt
from read_rainbow import get_rainbow, get_dqn
import matplotlib.patheffects as pe

games = ["BattleZone"]
files = ["BTR_adamw1_sqrt0_discount0997_lr_decay1_per1_taus64", "BTR_adamw1_sqrt0_discount0997_lr_decay1_per1_taus8",
         "BTR_adamw1_sqrt1_discount0997_lr_decay1_per1_taus64", "BTR_adamw0_sqrt0_ede0_discount0997_discount_anneal0_lr_decay1",
         "BTR_adamw1_sqrt0_ede0_discount0997", "BTR_adamw1_sqrt1_ede0_discount0997_discount_anneal0_lr_decay1"]

filenames = ["BTR", "BTR 8 Taus", "BTR + SQRT", "BTR 8 Taus, No WD", "BTR 8 Taus, No LRD", "BTR 8 Taus + SQRT","Rainbow", "DQN"]

"""files = ["BTR_discount099_avg", "BTR_adamw0_sqrt1_ede0_discount0997", "BTR_noisy0_spectral1_munch1_iqn0_double0", "BTR_noisy0_spectral1_munch0_iqn1_double0",
         "BTR_noisy1_spectral0_munch1_iqn1_dueling1_impala1_discount099", "BTR_noisy0_spectral1_munch1_iqn1_dueling0_impala1_discount099",
         "BTR_noisy0_spectral1_munch1_iqn1_dueling1_impala1_discount0997", "BTR_adamw1_sqrt0_ede0_discount0997", "BTR_adamw0_sqrt0_ede0_discount0997_discount_anneal1"]

filenames = ["BTR", "Discount+SQRT", "-IQN", "-Munchausen", "+Noisy -Spectral", "-Dueling", "Discount=0.997", "Discount+WeightDecay", "Discount0.97->0.997",
             "Rainbow*", "DQN"]"""

"""
files = ["BTR_adamw1_sqrt0_ede0_discount0997", "BTR_noisy0_spectral1_munch1_iqn0_double0", "BTR_discount099_avg",
         "BTR_noisy1_spectral0_munch1_iqn1_dueling1_impala1_discount099", ]

filenames = ["Beyond The Rainbow", "-IQN", "-Weight Decay", "-Spectral Normalization",
             "Rainbow*", "DQN"]
"""
combined_data = []
for file in files:
    # Load the combined data from the file
    combined_data.append(np.load("results\\" + file + "\\" + file + "BattleZoneEvaluation.npy")) # [:36] # add this for incomplete results
    print(np.load("results\\" + file + "\\" + file + "BattleZoneEvaluation.npy").shape)

combined_data = np.array(combined_data)
# data has shape (runs, 50 (evals_periods), 100 (eval_episodes))
print(combined_data.shape)

# Calculate the mean of each row (axis=2) to get the scores
mean_scores = np.mean(combined_data, axis=2)
print(mean_scores.shape)


mean_scores = np.vstack((mean_scores, get_rainbow(games[0])))
print(mean_scores.shape)

mean_scores = np.vstack((mean_scores, get_dqn(games[0])))
print(mean_scores.shape)

window_size = 5
# Create a rolling window function
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def add_first_scores(mean_data, smoothed_data):
    mean_data = mean_data[:, :5]

    # Create a cumulative sum along the second dimension
    cumulative_sum = np.cumsum(mean_data, axis=1)

    # Create a range of window sizes from 1 to 50
    window_sizes = np.arange(1, mean_data.shape[1] + 1)

    # Calculate the smoothed data by dividing the cumulative sum by the window size
    smoothed_data_start = cumulative_sum / window_sizes

    return np.concatenate((smoothed_data_start, smoothed_data), axis=1)


smoothed_scores = np.apply_along_axis(np.mean, axis=2, arr=rolling_window(mean_scores, window_size))
smoothed_scores = add_first_scores(mean_scores, smoothed_scores)
print(smoothed_scores.shape)



# Create an array for the X axis values (1 to 50)
#x_values = np.arange(1, total_frames)

# Create the plot
plt.figure(figsize=(10, 6))  # Set the figure size
for i in range(len(files) + 2):
    # Plot the mean scores against the X values
    if i == 0:
        plt.plot(smoothed_scores[i], linestyle='-', label=filenames[i], linewidth=3.,
                 path_effects=[pe.Stroke(linewidth=5, foreground='b'), pe.Normal()], color="gold")

    else:
        plt.plot(smoothed_scores[i], linestyle='-', label=filenames[i], linewidth=1.)

# Add labels and a title to the plot
plt.xlabel("Frames (M)")
plt.ylabel("Score")
plt.title("Scores on Atari BattleZone")

# Show the grid
plt.grid(True)
plt.legend()

# Show the plot
plt.show()

