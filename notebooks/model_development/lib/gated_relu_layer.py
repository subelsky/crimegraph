from matplotlib.pylab import f
import tensorflow as tf
keras = tf.keras

class GatedReluLayer(keras.layers.Layer):
    """
    num_gated_relu_hidden_units represents the dimensionality of the output space for the layer;
    controls how many features each node in the layer will have after transformation
    """
    def __init__(self, num_gated_relu_hidden_units=8, **kwargs):
        super(GatedReluLayer, self).__init__(**kwargs)

        self.num_gated_relu_hidden_units = num_gated_relu_hidden_units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.num_gated_relu_hidden_units),
                                 initializer='random_normal',
                                 trainable=True)

        self.b = self.add_weight(shape=(self.num_gated_relu_hidden_units,),
                                 initializer='zeros',
                                 trainable=True)

        self.u = self.add_weight(shape=(input_shape[-1], self.num_gated_relu_hidden_units),
                                 initializer='random_normal',
                                 trainable=True)

        self.c = self.add_weight(shape=(self.num_gated_relu_hidden_units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        relu_output = tf.nn.relu(tf.matmul(inputs, self.w) + self.b)
        gate = tf.nn.sigmoid(tf.matmul(inputs, self.u) + self.c)

        return relu_output * gate