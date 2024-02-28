import json
import numpy as np


def get_rainbow(game):
    # Path to the JSON file
    file_path = 'results\\' + game.lower() + '.json'

    # Load the data from the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Initialize a dictionary to store the sum and count of values for each iteration
    iteration_values = {}

    # Process each entry
    for entry in data:
        if entry['Agent'] == 'Rainbow' and 0 <= entry['Iteration'] <= 50:
            iteration = entry['Iteration']
            value = entry['Value']

            # If the iteration is not in the dictionary, initialize it
            if iteration not in iteration_values:
                iteration_values[iteration] = {'sum': 0, 'count': 0}

            # Accumulate the sum and count for this iteration
            iteration_values[iteration]['sum'] += value
            iteration_values[iteration]['count'] += 1

    # Calculate the average for each iteration and sort by iteration
    averaged_values = [iteration_values[i]['sum'] / iteration_values[i]['count'] for i in sorted(iteration_values)]

    # Convert the list of averages into a NumPy array
    values_array = np.array(averaged_values)
    values_array = np.delete(values_array, 0).reshape(1, 50)

    # Check the shape and the array
    print(values_array.shape)
    #print(values_array)
    return values_array

def get_dqn(game):
    # Path to the JSON file
    file_path = 'results\\' + game.lower() + '.json'

    # Load the data from the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Initialize a dictionary to store the sum and count of values for each iteration
    iteration_values = {}

    # Process each entry
    for entry in data:
        if entry['Agent'] == 'DQN' and 0 <= entry['Iteration'] <= 50:
            iteration = entry['Iteration']
            value = entry['Value']

            # If the iteration is not in the dictionary, initialize it
            if iteration not in iteration_values:
                iteration_values[iteration] = {'sum': 0, 'count': 0}

            # Accumulate the sum and count for this iteration
            iteration_values[iteration]['sum'] += value
            iteration_values[iteration]['count'] += 1

    # Calculate the average for each iteration and sort by iteration
    averaged_values = [iteration_values[i]['sum'] / iteration_values[i]['count'] for i in sorted(iteration_values)]

    # Convert the list of averages into a NumPy array
    values_array = np.array(averaged_values)
    values_array = np.delete(values_array, 0).reshape(1, 50)

    # Check the shape and the array
    print(values_array.shape)
    #print(values_array)
    return values_array


if __name__ == "__main__":
    get_rainbow("battlezone")