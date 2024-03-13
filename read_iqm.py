import numpy as np
import matplotlib.pyplot as plt
from read_rainbow import get_entry
import matplotlib.patheffects as pe
import scipy

human_scores = {"BattleZone": 37188, "DoubleDunk": -16, "NameThisGame": 8049, "Phoenix": 7243, "Qbert": 13455}
random_scores = {"BattleZone": 2360, "DoubleDunk": -19, "NameThisGame": 2292, "Phoenix": 761, "Qbert": 164}


games = ["BattleZone"]
frames = 40
files = ["ema0_C250"] # this doesn't include BTR and GameName
filenames = ["BTR EMA No WD or LRD"]

use_extra_algos = True

data = []
for file in files:
    temp = []
    for frame in range(frames):
        per_game_scores = []
        for game in games:
            # can remove the extra game part in np.load with new results
            filename = "BTR_" + game + str(frames) + "M_" + file
            new_file = np.load("results_final\\" + filename + "\\" + filename + game + "Evaluation.npy")[frame]
            mean_frame = np.mean(new_file, axis=0)

            human_norm = (mean_frame - random_scores[game]) / (human_scores[game] - random_scores[game])

            per_game_scores.append(human_norm)

        per_game_scores = np.array(per_game_scores)

        # calculate IQM
        per_game_scores = scipy.stats.trim_mean(np.array([per_game_scores]), 0.25, axis=None)
        #per_game_scores = np.mean(per_game_scores)

        temp.append(per_game_scores)


    data.append(temp[:])

data = np.array(data)
print("Data")
print(data.shape)
# Shape (files, frame)

max_frames_extra = 198
# add rainbow and other algorithms
if use_extra_algos:
    new_algos = ["Rainbow", "IQN", "DQN"]
    extra_algos = []
    for algo in new_algos:
        per_frame = []
        for frame in range(max_frames_extra):
            per_game_scores = []
            for game in games:
                new_entry = get_entry(game, algo, max_frames_extra)[:, frame]

                new_entry = (new_entry - random_scores[game]) / (human_scores[game] - random_scores[game])

                per_game_scores.append(new_entry)

            per_game_scores = np.array(per_game_scores)
            per_game_scores = scipy.stats.trim_mean(np.array([per_game_scores]), 0.25, axis=None)
            per_frame.append(per_game_scores)

        extra_algos.append(per_frame)

    print("extra_algos")
    extra_algos = np.array(extra_algos)
    print(extra_algos.shape)

#data = np.vstack((data, extra_algos))

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

# plot graphs

plt.figure(figsize=(10, 6))  # Set the figure size

for i in range(len(data)):
    # Plot the mean scores against the X values
    if i == 0:
        plt.plot(smoothed_scores[i], linestyle='-', label=filenames[i], linewidth=4.,
                 path_effects=[pe.Stroke(linewidth=6, foreground='b'), pe.Normal()], color="gold")

    else:
        plt.plot(smoothed_scores[i], linestyle='-', label=filenames[i], linewidth=2.)

colors = ["blue", "red", "purple"]
if use_extra_algos:

    for i in range(len(extra_algos)):
        # Plot the mean scores against the X values

        x = extra_algos[i]
        print(x.shape)
        y = np.arange(0, len(extra_algos[0]))

        x1, y1 = x[:frames], y[:frames]  # Data for the first part
        x2, y2 = x[frames - 1:], y[frames - 1:]

        print(x1.shape)
        print(x2.shape)

        plt.plot(y1, x1, linestyle='-', label=new_algos[i], linewidth=2., color=colors[i])
        plt.plot(y2, x2, linestyle=':', linewidth=2., color=colors[i])


# Add labels and a title to the plot
plt.xlabel("Frames (M)")
plt.ylabel("IQM Score")

# Show the grid
plt.grid(True)
plt.legend()

# Show the plot
plt.show()


