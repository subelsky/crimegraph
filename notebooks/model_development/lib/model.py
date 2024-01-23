import tensorflow as tf
keras = tf.keras

from keras.layers import TimeDistributed, Lambda, Flatten, Sequential, Dense

from lib.gated_relu_layer import GatedReluLayer
from lib.localized_diffusion_network import LocalizedDiffusionNetwork
from lib.loss_function import create_weighted_mse_loss
from lib.optimizer import build_optimizer

def build_model(adjacency_matrix, metrics, loss_rho=0.01, num_gated_relu_layers=3, **kwargs):
    num_nodes = len(adjacency_matrix)

    model = Sequential(name='GLDNet')

    for i in range(num_gated_relu_layers):
        model.add(GatedReluLayer(name=f'GatedRelu{i}', **kwargs))

    localized_diffusion_network = LocalizedDiffusionNetwork(transition_matrix=adjacency_matrix, **kwargs)
    model.add(localized_diffusion_network)

    # batch_norm_layer = tf.keras.layers.BatchNormalization()
    # model.add(batch_norm_layer)

    # Aggregating the features for each node into a single value using a Dense layer
    # This will change the output shape to (None, 7, 766, 1)
    model.add(TimeDistributed(Dense(units=1, activation='sigmoid'), name='FeatureAggregation'))

    # We want only the prediction for the last timestep, so we select the last timestep in the output
    # This will change the output shape to (None, 766, 1)
    model.add(Lambda(lambda x: x[:, -1, :, 0], name='SelectLastTimestep'))

    optimizer = build_optimizer(**kwargs)
    loss_function = create_weighted_mse_loss(loss_rho)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    return model

def build_linear_model(metrics, lookback_steps=7, num_nodes=766, num_features=3, total_num_timesteps=3225, **kwargs):
    loss_rho = kwargs.get('loss_rho', 0.01)

    linear_model = Sequential([
        # Flatten the input to remove the temporal structure (for simplicity)
        Flatten(input_shape=(lookback_steps, num_nodes, num_features), name='FlattenInput'),
        
        # Since we're predicting for each node, the output dimension is num_nodes
        Dense(num_nodes, name='DenseOutput')
    ], name='LinearModel')

    optimizer = build_optimizer(**kwargs)
    loss_function = create_weighted_mse_loss(loss_rho)

    linear_model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    
    return linear_model

class NaiveModel(keras.Model):
    def call(self, inputs):
        # Extract the most recent value of the last feature column of the last timestep
        # inputs shape is assumed to be (batch_size, lookback_steps, num_nodes, num_features)
        # Returning the last timestep and the last feature
        return inputs[:, -1, :, -1]

def build_naive_model(metrics, **kwargs):
    naive_model = NaiveModel(name='NaiveModel')
    optimizer = build_optimizer(**kwargs)

    naive_model.compile(optimizer=optimizer, metrics=metrics)
    
    return naive_model