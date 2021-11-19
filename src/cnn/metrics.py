import tensorflow as tf


class CustomAUC(tf.keras.metrics.Metric):

    def __init__(self, n_splits, name='AUC', **kwargs):
        super().__init__(name=name, **kwargs)
        self.auc = tf.keras.metrics.AUC()
        self.n_splits = n_splits

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.split(y_true, self.n_splits, axis=-1)[0]
        y_pred = tf.split(y_pred, self.n_splits, axis=-1)[0]
        self.auc.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.auc.result()

    def reset_states(self):
        self.auc.reset_states()


# CustomMRE (for peak area)
