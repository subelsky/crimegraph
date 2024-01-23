import tensorflow as tf

class HitRateMetric(tf.keras.metrics.Metric):
    def __init__(self, num_nodes, coverage, name='hit_rate', **kwargs):
        super(HitRateMetric, self).__init__(name=name, **kwargs)

        self.num_nodes = num_nodes
        self.coverage = coverage
        self.num_top_nodes = tf.cast(self.coverage * tf.cast(self.num_nodes, tf.float32), tf.int32)
        self.total_hit_rate = self.add_weight(name='total_hit_rate', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # For each example in the batch, compute the hit rate
        batch_hit_rates = tf.map_fn(lambda x: self.calculate_hit_rate_for_example(x[0], x[1]), 
                                    (y_true, y_pred), fn_output_signature=tf.float32)

        batch_mean_hit_rate = tf.reduce_mean(batch_hit_rates)

        self.total_hit_rate.assign_add(batch_mean_hit_rate)
        self.count.assign_add(1)

    def result(self):
        return self.total_hit_rate / self.count

    def reset_state(self):
        self.total_hit_rate.assign(0.0)
        self.count.assign(0.0)

    def calculate_hit_rate_for_example(self, y_true_example, y_pred_example):
        # Identify top predicted nodes based on coverage
        top_pred_indices = tf.argsort(y_pred_example, direction='DESCENDING')[:self.num_top_nodes]

        # Create a mask for the top predicted nodes
        top_pred_mask = tf.reduce_any(tf.equal(tf.expand_dims(top_pred_indices, -1), 
                                              tf.range(self.num_nodes)), axis=0)

        # Calculate the hit rate
        hit_rate = tf.reduce_sum(tf.cast(tf.logical_and(top_pred_mask, y_true_example > 0), 
                                         tf.float32)) / tf.reduce_sum(tf.cast(y_true_example > 0, tf.float32))

        return hit_rate