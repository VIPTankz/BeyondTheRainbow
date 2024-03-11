import numpy as np
import matplotlib.pyplot as plt

games = ["BattleZone", "Qbert"]
frames = 40
files = ["ema1_C8000"]
filenames = ["BTR EMA No WD or LRD"]

data = []
for file in files:
    temp = []
    for frame in range(frames):
        per_game_scores = []
        for game in games:
            # can remove the extra game part in np.load with new results
            filename = "BTR_" + game + str(frames) + "M_" + file
            new_file = np.load("results_final\\" + filename + "\\" + filename + game + "ParamNorms.npy")[frame]

            per_game_scores.append(new_file)

        per_game_scores = np.array(per_game_scores)

        # calculate Mean
        per_game_scores = np.mean(per_game_scores)

        temp.append(per_game_scores)


    data.append(temp[:])

data = np.array(data)
print(data.shape)

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

# apply rolling average
smoothed_scores = np.apply_along_axis(np.mean, axis=2, arr=rolling_window(data, window_size))

# apply start bias
smoothed_scores = add_first_scores(data, smoothed_scores)
print("Final Data Shape")
print(smoothed_scores.shape)

plt.figure(figsize=(10, 6))  # Set the figure size

for i in range(len(smoothed_scores)):
    # Plot the mean scores against the X values
    plt.plot(smoothed_scores[i], linestyle='-', label=filenames[i], linewidth=2.)


# Add labels and a title to the plot
plt.xlabel("Frames (M)")
plt.ylabel("Average Weight Norm")

# Show the grid
plt.grid(True)
plt.legend()

# Show the plot
plt.show()