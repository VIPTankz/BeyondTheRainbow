import numpy as np
import matplotlib.pyplot as plt

games = ["BattleZone"]
files = ["BTR_ema_tau25e-4", "BTR_noisy0_spectral1_munch1_iqn0_double0", "BTR_noisy0_spectral1_munch0_iqn1_double0",
         "BTR_noisy1_spectral0_munch1_iqn1_dueling1_impala1_discount099"]

filenames = ["BTR", "-IQN", "-Munchausen", "+Noisy -Spectral"]

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

window_size = 5
# Create a rolling window function
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def add_first_scores(mean_data, smoothed_data):
    print(mean_data.shape)
    mean_data = mean_data[:, :5]
    print(mean_data.shape)

    # Create a cumulative sum along the second dimension
    cumulative_sum = np.cumsum(mean_data, axis=1)

    # Create a range of window sizes from 1 to 50
    window_sizes = np.arange(1, mean_data.shape[1] + 1)

    # Calculate the smoothed data by dividing the cumulative sum by the window size
    smoothed_data_start = cumulative_sum / window_sizes

    print(smoothed_data.shape)
    print(mean_data.shape)

    return np.concatenate((smoothed_data_start, smoothed_data), axis=1)


smoothed_scores = np.apply_along_axis(np.mean, axis=2, arr=rolling_window(mean_scores, window_size))
smoothed_scores = add_first_scores(mean_scores, smoothed_scores)
print(smoothed_scores.shape)



# Create an array for the X axis values (1 to 50)
#x_values = np.arange(1, total_frames)

# Create the plot
plt.figure(figsize=(10, 6))  # Set the figure size
for i in range(len(files)):
    # Plot the mean scores against the X values
    plt.plot(smoothed_scores[i], linestyle='-', label=filenames[i])

# Add labels and a title to the plot
plt.xlabel("Frames (M)")
plt.ylabel("Score")
plt.title("Scores on Atari BattleZone")

# Show the grid
plt.grid(True)
plt.legend()

# Show the plot
plt.show()

