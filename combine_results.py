
import numpy as np

# Initialize an empty array to store the combined data
combined_data = []

# Loop through the range of X values (from 1 million to 50 million)
for X in range(1000000, 51000000, 1000000):
    # Load each file and append it to the combined_data list
    file_name = f"results\\BTR_og_old_maybe_delete\\BTR_ogBattleZoneEvaluation{X}.npy"
    data = np.load(file_name)
    combined_data.append(data)
    print(data)

# Convert the list of arrays into a single numpy array
combined_data = np.array(combined_data)

# Verify the shape of the combined data (should be (50, 100))
print(combined_data.shape)

# Save the combined data to a new file
np.save("BTR_ogBattleZoneEvaluation.npy", combined_data)
