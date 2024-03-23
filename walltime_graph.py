import matplotlib.pyplot as plt

# Given data
values = [56, 58, 65, 67, 87, 100, 170, 325]
labels = ["Rainbow DQN", "BTR -IQN", "BTR -Munchausen", "BTR -IMPALA", "BTR -Spectral Normalization",
          "BTR", "BTR -Maxpooling", "BTR -Vectorized Environments"]

colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', "magenta", "gold"]

# Creating a sideways bar chart
plt.figure(figsize=[10, 6])  # Set the figure size
bars = plt.barh(labels, values, color=colors)  # Create horizontal bars
plt.xlabel('Relative Walltime (Lower is Better)')  # Label for the x-axis
#plt.title('Walltime For different Components and Algorithms')  # Title of the chart
plt.gca().invert_yaxis()  # Invert the y-axis so the highest value is on top

for bar, value in zip(bars, values):
    # Calculate the percentage difference
    percentage_diff = (value - 100)
    # Format the text with the calculated percentage
    if percentage_diff < 0:
        text = f' {percentage_diff:.0f}%' if percentage_diff != 0 else ' +0%'
    else:
        text = f' +{percentage_diff:.0f}%' if percentage_diff != 0 else ' +0%'
    # Place the text to the right of the bar
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, text,
             va='center', ha='left')  # Adjusts the position of the text

plt.show()  # Display the chart