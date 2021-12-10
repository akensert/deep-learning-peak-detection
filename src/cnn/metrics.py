import tensorflow as tf



class CustomTruePositiveRate(tf.keras.metrics.Metric):

    def __init__(self, threshold, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_negatives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.split(y_true, 3, axis=-1)[0]
        y_pred = tf.split(y_pred, 3, axis=-1)[0]

        y_pred = y_pred > self.threshold
        y_true = tf.cast(y_true, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, dtype=self.dtype)
        self.true_positives.assign_add(tf.reduce_sum(values))

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        values = tf.cast(values, dtype=self.dtype)
        self.false_negatives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives / (self.true_positives + self.false_negatives)

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_negatives.assign(0)


class CustomFalsePositiveRate(tf.keras.metrics.Metric):

    def __init__(self, threshold, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.false_positives = self.add_weight(name='tp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.split(y_true, 3, axis=-1)[0]
        y_pred = tf.split(y_pred, 3, axis=-1)[0]

        y_pred = y_pred > self.threshold
        y_true = tf.cast(y_true, tf.bool)

        values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        values = tf.cast(values, dtype=self.dtype)
        self.false_positives.assign_add(tf.reduce_sum(values))

        values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
        values = tf.cast(values, dtype=self.dtype)
        self.true_negatives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.false_positives / (self.false_positives + self.true_negatives)

    def reset_states(self):
        self.false_positives.assign(0)
        self.true_negatives.assign(0)


class CustomAUC(tf.keras.metrics.AUC):

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.split(y_true, 3, axis=-1)[0]
        y_pred = tf.split(y_pred, 3, axis=-1)[0]

        super().update_state(y_true, y_pred, sample_weight)


class CustomAccuracy(tf.keras.metrics.BinaryAccuracy):

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.split(y_true, 3, axis=-1)[0]
        y_pred = tf.split(y_pred, 3, axis=-1)[0]

        super().update_state(y_true, y_pred, sample_weight)


class CustomMREArea(tf.keras.metrics.Mean):

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_prob, _, y_true = tf.split(y_true, 3, axis=-1)
        y_pred_prob, _, y_pred = tf.split(y_pred, 3, axis=-1)

        mask = tf.math.logical_and(
            tf.math.equal(y_true_prob, 1.), tf.math.greater(y_pred_prob, 0.5))

        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        error = tf.math.divide_no_nan(
            tf.math.abs(y_true - y_pred),
            tf.math.abs(y_true)
        )
        super().update_state(error, sample_weight=sample_weight)


class CustomMRELoc(tf.keras.metrics.Mean):

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_prob, y_true, _ = tf.split(y_true, 3, axis=-1)
        y_pred_prob, y_pred, _  = tf.split(y_pred, 3, axis=-1)

        mask = tf.math.logical_and(
            tf.math.equal(y_true_prob, 1.), tf.math.greater(y_pred_prob, 0.5))

        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        error = tf.math.divide_no_nan(
            tf.math.abs(y_true - y_pred),
            tf.math.abs(y_true)
        )
        super().update_state(error, sample_weight=sample_weight)


class CustomMAELoc(tf.keras.metrics.Mean):

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_prob, y_true, _ = tf.split(y_true, 3, axis=-1)
        y_pred_prob, y_pred, _  = tf.split(y_pred, 3, axis=-1)

        mask = tf.math.logical_and(
            tf.math.equal(y_true_prob, 1.), tf.math.greater(y_pred_prob, 0.5))

        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        error = tf.math.abs(y_true - y_pred)

        super().update_state(error, sample_weight=sample_weight)


def get_accuracy_metrics_at_thresholds(
    thresholds=[
        # has been narrowed down from [0.05, 0.95]
        0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55
    ]
):
    return [
        CustomAccuracy(name='acc_' + str(t).split('.')[-1], threshold=t)
        for t in thresholds
    ]
