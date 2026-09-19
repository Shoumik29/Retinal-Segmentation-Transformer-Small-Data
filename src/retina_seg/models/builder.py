"""
Responsibility
    - Builds the model from config.

This file has almost no logic of its own. It composes two registered
stages (encoder and decoder) by name from cfg.model.
"""


from tensorflow.keras import Model, layers
from . import decoders, encoders  # This import is for populating the corresponding registry
from .registry import DECODERS, ENCODERS


def build_model(cfg) -> Model:
 
    """
    Build a model from config.

    Args:
        loaded config with:
            cfg.model.img_size, 
            cfg.model.num_classes
            cfg.model.encoder
            cfg.model.decoder
            cfg.model.encoder_kwargs 
            cfg.model.decoder_kwargs (dicts for extra keyword arguments with the encoder/decoder function)

    Returns:
        An uncompiled tf.keras.Model.

    Raises:
        KeyError: if cfg.model.encoder/decoder names a component that
            isn't registered. The error lists what is available.
    """

    encoder_fn = ENCODERS.get(cfg.model.encoder)
    decoder_fn = DECODERS.get(cfg.model.decoder)

    encoder_kwargs = dict(cfg.model.get("encoder_kwargs", {}))
    decoder_kwargs = dict(cfg.model.get("decoder_kwargs", {}))

    input_shape = (cfg.model.img_size, cfg.model.img_size, 3)
    inputs = layers.Input(shape=input_shape, name="input_image")

    feature_maps = list(encoder_fn(inputs, **encoder_kwargs))
    feature_maps.reverse()

    decoder_output = decoder_fn(feature_maps, **decoder_kwargs)
    outputs = layers.Dense(cfg.model.num_classes, activation="softmax")(decoder_output)

    return Model(inputs=inputs, outputs=outputs)