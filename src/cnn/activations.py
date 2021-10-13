import tensorflow as tf


class CustomActivation(tf.keras.layers.Layer):

    def __init__(self, n_splits, **kwargs):
        super().__init__(**kwargs)
        self.n_splits = n_splits

    def call(self, inputs):
        pred, loc, area = tf.split(inputs, self.n_splits, axis=-1)
        pred = tf.nn.sigmoid(pred)
        loc = tf.nn.sigmoid(loc)
        return tf.concat([pred, loc, area], axis=-1)
