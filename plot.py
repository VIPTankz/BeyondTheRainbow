import numpy as np
import matplotlib.pyplot as plt
runs = 1

data_files = ["MultipleTracks21"]

expers = []

for exper in data_files:
    temp = []
    for i in range(runs):
        temp.append(np.load(exper + '.npy'))
    expers.append(temp[:])

# Example 2D list
data = list(expers[0][0])
print(len(data))

# Number of data entries to average
average_size = 1000

# Extracting scores and steps from the data
scores = [row[0] for row in data]
steps = [row[2] for row in data]

steps = [x / 3600 for x in steps]


# Averaging scores over a given number of data entries
averaged_scores = []
for i in range(len(scores) - average_size + 1):
    average = sum(scores[i:i+average_size]) / average_size
    averaged_scores.append(average)

# Creating the plot
plt.plot(steps[:len(averaged_scores)], averaged_scores)
plt.xlabel('Hours')
plt.ylabel('Average Reward')
plt.title('Super Mario Galaxy')
plt.grid(True)

# Displaying the plot
plt.show()
print("done?")
