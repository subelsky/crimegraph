from keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import Adam, SGD, RMSprop
import platform
import tensorflow as tf
keras = tf.keras

def build_optimizer(optimizer_name='Adam', initial_learning_rate=0.03, decay_steps=10000, decay_rate=0.9, staircase=True, **kwargs):
    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase
    )

    if optimizer_name == 'Adam':
        if platform.system() == "Darwin" and platform.processor() == "arm":
            optimizer_class = keras.optimizers.legacy.Adam
        else:
            optimizer_class = Adam
    elif optimizer_name == 'SGD':
        optimizer_class = SGD
    elif optimizer_name == 'RMSprop':
        optimizer_class = RMSprop
    else:
        raise ValueError(f"'{optimizer_name}' not supported")

    optimizer = optimizer_class(learning_rate=lr_schedule)

    return optimizer
