import tensorflow as tf
from tensorflow.keras import layers

from .activations import CustomActivation


def conv_block(
    inputs,
    filters,
    kernel_size,
    dropout=0.0,
    num_conv_layers=1,
    pool_type='conv',
    pool_size=2,
    residual=False,
):

    x = inputs
    for i in range(num_conv_layers):

        if str(pool_type).lower().startswith('conv') and i == num_conv_layers-1:
            x = layers.Conv1D(
                filters, kernel_size, strides=pool_size, padding='same')(x)
        else:
            x = layers.Conv1D(
                filters, kernel_size, strides=1, padding='same')(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        if isinstance(dropout, float):
            x = layers.SpatialDropout1D(dropout)(x)

    if str(pool_type).lower().startswith('max'):
        x = layers.MaxPool1D(pool_size)(x)
    elif str(pool_type).lower().startswith(('mean', 'average', 'avg')):
        x = layers.AvgPool1D(pool_size)(x)

    if residual:
        r = layers.Conv1D(filters, kernel_size, strides=pool_size, padding='same')(inputs)
        r = layers.BatchNormalization()(r)
        r = layers.Activation('relu')(r)
        x = layers.Add()([x, r])

    if isinstance(dropout, float):
        x = layers.SpatialDropout1D(dropout)(x)

    return x

def ConvNet(
    filters=[64, 128, 256],
    kernel_sizes=[9, 9, 9],
    dropout=0.0,
    pool_type='conv',
    pool_sizes=[4, 4, 2],
    conv_block_size=1,
    input_shape=(8192, 1),
    output_shape=(256, 3),
    residual=False,
):

    if input_shape[0] % output_shape[0] != 0:
        raise ValueError(
            "Input length is not a multiple of output length: " +
            f"{input_shape[0]} vs. {output_shape[0]}"
        )

    inputs = layers.Input(input_shape)

    x = inputs
    for i, (filt, kernel_size, pool_size) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
        x = conv_block(x, filt, kernel_size,
                       dropout=dropout,
                       num_conv_layers=conv_block_size,
                       pool_type=pool_type,
                       pool_size=pool_size,
                       residual=True)

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
