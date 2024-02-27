import numpy as np

# List of file names
files = ["BTR_noisy0_spectral1_munch1_iqn1_dueling1_impala1_discount099",
         "BTR_ema_tau25e-4"]

# Preallocate list to store arrays
arrays = []

# Load each file and append to the list
for file in files:
    array = np.load("results\\" + file + "\\" + file + "BattleZoneEvaluation.npy")
    arrays.append(array)

# Compute the average along the specified dimension
average_array = np.mean(np.array(arrays), axis=0)

# Save the averaged array
np.save("results\\averaged_BattleZoneEvaluation.npy", average_array)
