import tensorflow as tf
keras = tf.keras
from keras.layers import Layer
import numpy as np

class LocalizedDiffusionLayer(Layer):
    def __init__(self, units, adjacency_matrix, **kwargs):
        super(LocalizedDiffusionLayer, self).__init__(**kwargs)
        self.units = units
        self.adjacency_matrix = tf.cast(adjacency_matrix, dtype=tf.float32)

    def build(self, input_shape):
        self.kernel_self = self.add_weight(name='kernel_self', 
                                           shape=(input_shape[-1], self.units),
                                           initializer='uniform',
                                           trainable=True)

        self.kernel_local = self.add_weight(name='kernel_local',
                                            shape=(input_shape[-1], self.units),
                                            initializer='uniform',
                                            trainable=True)

    def call(self, inputs):
        # Apply self-connection weights
        output = tf.matmul(inputs, self.kernel_self)

        # Perform diffusion
        step_output = tf.matmul(self.adjacency_matrix, inputs)

        # Apply localized weights to the diffused values
        step_output = tf.matmul(step_output, self.kernel_local)
        output += step_output

        return tf.nn.relu(output)

    def get_config(self):
        config = super(LocalizedDiffusionLayer, self).get_config()
        config.update({ 'units': self.units })

        return config

class LocalizedDiffusionNetwork(tf.keras.Model):
    def __init__(self, transition_matrix, num_ldn_layers=5, ldn_hidden_units_per_layer=4, **kwargs):
        super(LocalizedDiffusionNetwork, self).__init__(name=f'{num_ldn_layers}LayerLDN', **kwargs)

        self.ldn_layers = [
            LocalizedDiffusionLayer(units=ldn_hidden_units_per_layer, adjacency_matrix=transition_matrix)
            for i in range(num_ldn_layers)
        ]

    def call(self, inputs):
        x = inputs

        for layer in self.ldn_layers:
            x = layer(x)

        return x