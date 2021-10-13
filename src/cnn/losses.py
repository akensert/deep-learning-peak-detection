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
        bce_loss_1 = tf.keras.losses.BinaryCrossentropy()(true_prob, pred_prob)
        # Compute loss only for instances in mask
        bce_loss_2 = tf.keras.losses.BinaryCrossentropy()(
            tf.boolean_mask(true_loc, mask), tf.boolean_mask(pred_loc, mask))
        # Compute loss only for instances in mask
        huber_loss_1 = tf.keras.losses.Huber()(
            tf.boolean_mask(true_area, mask), tf.boolean_mask(pred_area, mask))

        return (
            bce_loss_1 * self.weight_prob +
            bce_loss_2 * self.weight_loc +
            huber_loss_1 * self.weight_area
        )
