import numpy as np
import matplotlib.pyplot as plt

file = "SpeedTests_envs64_bs256"
total_frames = 8 + 1

# Load the combined data from the file
combined_data = np.load("results\\" + file + "\\" + file + "BattleZoneEvaluation.npy")
print(combined_data.shape)

# Calculate the mean of each row (axis=1) to get the scores
mean_scores = np.mean(combined_data, axis=1)

# Create an array for the X axis values (1 to 50)
x_values = np.arange(1, total_frames)

# Create the plot
plt.figure(figsize=(10, 6))  # Set the figure size

# Plot the mean scores against the X values
plt.plot(x_values, mean_scores, marker='o', linestyle='-')

# Add labels and a title to the plot
plt.xlabel("Frames (M)")
plt.ylabel("Score")
plt.title("Mean Score vs. Frames")

# Show the grid
plt.grid(True)

# Show the plot
plt.show()

