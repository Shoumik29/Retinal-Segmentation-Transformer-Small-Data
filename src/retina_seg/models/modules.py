"""
Responsibility
    - Define modules used across multiple encoders/decoders.
"""


from tensorflow.keras import layers


def conv_block(x, num_filters, kernel):
    x = layers.Conv2D(filters=num_filters, kernel_size=(kernel, kernel), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(0.3)(x)
    return x
