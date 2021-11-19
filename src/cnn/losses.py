import tensorflow as tf


class CustomLoss(tf.keras.losses.Loss):

    def __init__(
        self,
        n_splits,
        weight_prob=1.0,
        weight_loc=1.0,
        weight_area=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_splits = n_splits
        self.weight_prob = weight_prob
        self.weight_loc = weight_loc
        self.weight_area = weight_area

    def call(self, y_true, y_pred, sample_weight=None):

        # Cast to dtype float32 just in case
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Unpack trues and preds
        pred_prob, pred_loc, pred_area = tf.split(y_pred, self.n_splits, axis=-1)
        true_prob, true_loc, true_area = tf.split(y_true, self.n_splits, axis=-1)

        # Mask for where y_true is indicating a peak
        mask = tf.math.equal(true_prob, 1.)

        # Compute loss for all instances
        prob_loss = tf.keras.losses.BinaryCrossentropy()(true_prob, pred_prob)
        # Compute loss only for instances in mask
        loc_loss = tf.keras.losses.BinaryCrossentropy()(
            tf.boolean_mask(true_loc, mask), tf.boolean_mask(pred_loc, mask))
        area_loss = MeanRelativeError()(
            tf.boolean_mask(true_area, mask), tf.boolean_mask(pred_area, mask))

        return (
            prob_loss * self.weight_prob +
            loc_loss * self.weight_loc +
            area_loss * self.weight_area
        )

class MeanRelativeError(tf.keras.losses.Loss):

    def call(self, y_true, y_pred, sample_weight=None):
        return tf.math.abs(y_true - y_pred) / y_true
