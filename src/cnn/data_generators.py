import numpy as np
import tensorflow as tf
import math


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(
        self,
        indices,
        simulator,
        label_encoder,
        batch_size=32,
        shuffle=False
    ):
        self.indices = indices
        self.simulator = simulator
        self.label_encoder = label_encoder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size: (1 + index) * self.batch_size
        ]
        x_batch, y_batch = [], []
        for data in self.simulator.sample_batch(batch_indices):
            y = self.label_encoder.encode(data['loc'], data['area'])
            x = data['chromatogram'][:, None]
            x_batch.append(x)
            y_batch.append(y)

        return np.array(x_batch), np.array(y_batch)
