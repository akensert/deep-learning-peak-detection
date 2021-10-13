import tensorflow as tf
from tensorflow.keras import layers

from .activations import CustomActivation


def conv_block(x, filters, kernel_size, activation='relu', dropout=0.1,
               num_conv_layers=1, pool_type='max', batch_norm=True):

    for _ in range(num_conv_layers):
        x = layers.Conv1D(filters, kernel_size, padding='same')(x)
        x = layers.Activation(activation)(x)

        if batch_norm:
            x = layers.BatchNormalization()(x)

        if isinstance(dropout, float):
            x = layers.SpatialDropout1D(dropout)(x)

    if pool_type is None:
        return x

    if pool_type.lower().startswith('max'):
        x = layers.MaxPool1D()(x)
    elif pool_type.lower().startswith(('mean', 'average', 'avg')):
        x = layers.AveragePool1D()(x)

    return x


def ConvNet(
    filters=[128, 128, 256, 256, 512],
    kernel_sizes=[9, 9, 9, 9, 9],
    pool_type='max',
    input_shape=(16384, 1),
    output_shape=(512, 3)
):

    if input_shape[0] % output_shape[0] != 0:
        raise ValueError(
            "Input length is not a multiple of output length: " +
            f"{input_shape[0]} vs. {output_shape[0]}"
        )

    inputs = layers.Input(input_shape)

    x = inputs
    for i, (filt, kernel_size) in enumerate(zip(filters, kernel_sizes)):
        x = conv_block(x, filt, kernel_size, num_conv_layers=1, pool_type=pool_type)

        if x.shape[1] < output_shape[0]:
            raise ValueError(
                "Length of downstream feature maps is smaller " +
                f"than the output length: {x.shape[1]} vs. {output_shape[0]}. " +
                "To fix this, reduce the number of layers."
            )

    x = conv_block(x, filters[-1], 1, pool_type=None)

    x = layers.Conv1D(output_shape[1], kernel_size=1, strides=1)(x)

    outputs = CustomActivation(output_shape[1])(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
