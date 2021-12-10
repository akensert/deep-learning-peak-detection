import tensorflow as tf


class SaveModelWeightsCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(f'output/weights/weights_{epoch:03d}.h5')
