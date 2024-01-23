import tensorflow as tf

def create_weighted_mse_loss(rho=0.01):
    def weighted_mse_loss(y_true, y_pred):
        """
        Weighted Mean Squared Error Loss Function to prioritize rarer events
        """

        # Weights for each sample: use rho for non-events, and the true value for events
        weights = tf.where(tf.greater(y_true, 0), y_true, rho * tf.ones_like(y_true))

        # Calculate the squared error, weighted by the weights tensor
        squared_error = tf.square(y_pred - y_true)
        weighted_squared_error = weights * squared_error

        # Calculate the mean weighted squared error across all samples
        loss = tf.reduce_mean(weighted_squared_error)

        return loss
    
    return weighted_mse_loss