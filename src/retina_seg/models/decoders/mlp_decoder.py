"""
Responsibility
	- Define Experimental Decoders.
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from ..modules import (
	conv_block
)


def LFE(feature_maps):	

	projection = 256
	
	x = feature_maps[-1]
	
	x = layers.Reshape((int(x.shape[1] ** (1/2)), int(x.shape[1] ** (1/2)), x.shape[-1]))(x)
	
	# Compression
	x = conv_block(x, projection, 3)
	
	x1 = tf.roll(x, shift=1, axis=1)
	x2 = tf.roll(x, shift=-1, axis=1)
	x3 = layers.Concatenate(axis=3)([x, x1, x2])
	x3 = conv_block(x3, projection, 3)
	
	x4 = tf.roll(x, shift=1, axis=2)
	x5 = tf.roll(x, shift=-1, axis=2)
	x6 = layers.Concatenate(axis=3)([x, x4, x5])
	x6 = conv_block(x6, projection, 3)
	
	x = layers.Add(name='feature_grid')([x3, x6])

	fused_feature = x	
	
	x = fused_feature
	x = layers.Conv2DTranspose(96, kernel_size = (2, 2), strides = (2, 2), padding = "valid")(x)
	x = conv_block(x, 96, 3)
	
	x = layers.Conv2DTranspose(48, kernel_size = (2, 2), strides = (2, 2), padding = "valid")(x)
	x = conv_block(x, 48, 3)
	
	return x
	