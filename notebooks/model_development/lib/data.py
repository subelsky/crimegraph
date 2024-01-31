import numpy as np
import pandas as pd
import tensorflow as tf

# Exclude the columns 'Arrest' and 'Property' for now, although we may want to use them later
# if we experiment with new exponential smoothing factors during hyperparameter tuning
FEATURE_COLUMNS = ['ArrestSmoothed', 'PropertySmoothed', 'ViolentSmoothed']

# The label column is what we want the model to predict, one day in the future
LABEL_COLUMN = 'Violent'

def load_timesteps(file_path):
    # Load the timesteps dataframe, ensuring that integer columns are read as integers
    timesteps = pd.read_csv(file_path, dtype={'NodeIndex': 'int', 'Arrest': 'int', 'Property': 'int', 'Violent': 'int'})

    # Groups the timesteps by date and node index so we can construct a tensor representing timesteps x num_nodes x features
    timesteps.reset_index(inplace=True)
    timesteps.set_index(['Date', 'NodeIndex'], inplace=True)

    # Filter the timesteps DataFrame to keep only the feature columns and label
    timesteps = timesteps[FEATURE_COLUMNS + [LABEL_COLUMN]]

    return timesteps

def create_data_tensor(timesteps, num_nodes):
    # Identify unique dates and nodes
    unique_dates = timesteps.index.get_level_values('Date').unique()
    all_node_indices = np.arange(num_nodes)

    # Initialize a 3D tensor with zeros: shape (num_dates, num_nodes, num_features + label)
    # +1 in the features dimension to accommodate the label
    tensor_shape = (len(unique_dates), len(all_node_indices), len(FEATURE_COLUMNS) + 1)
    data_tensor = np.zeros(tensor_shape)

    # Populate the tensor with values from the DataFrame
    for date_idx, date in enumerate(unique_dates):
        for node_idx, node in enumerate(all_node_indices):
            if (date, node) in timesteps.index:
                # Extract the features and label for the current date and node
                data = timesteps.loc[(date, node)].values
                data_tensor[date_idx, node_idx, :] = data
    
    return data_tensor

def create_dataset(data_tensor_slice, lookback_timesteps=7):
    num_steps, num_nodes, num_features = data_tensor_slice.shape

    inputs = np.zeros((num_steps - lookback_timesteps, lookback_timesteps, num_nodes, num_features - 1))
    labels = np.zeros((num_steps - lookback_timesteps, num_nodes))

    for t in range(lookback_timesteps, num_steps):
        inputs[t - lookback_timesteps] = data_tensor_slice[t - lookback_timesteps:t, :, :-1]
        labels[t - lookback_timesteps] = data_tensor_slice[t, :, -1]

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

    return dataset