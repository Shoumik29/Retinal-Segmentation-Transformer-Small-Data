import pytest
import tensorflow as tf
from retina_seg.models.builder import build_model


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

class ModelConfig:
    class Model:
        img_size = 224
        num_classes = 2
        encoder = "swin_transformer"
        decoder = "lfe"

        encoder_kwargs = {}
        decoder_kwargs = {}

        def get(self, key, default=None):
            return getattr(self, key, default)

    model = Model()


@pytest.fixture
def model():
    """Build a fresh model for each test."""
    return build_model(ModelConfig())


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_model_can_be_built(model):
    """The configured encoder and decoder should produce a valid Keras model."""
    assert isinstance(model, tf.keras.Model)


def test_model_has_expected_input_shape(model):
    """The model should accept RGB images of the configured resolution."""
    assert model.input_shape == (None, 224, 224, 3)


def test_model_has_expected_number_of_classes(model):
    """The final prediction layer should contain one channel per class."""
    assert model.output_shape[-1] == 2


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_model_forward_pass(model):
    """The model should successfully process a batch of RGB images."""

    x = tf.random.normal((2, 224, 224, 3))

    y = model(x, training=False)

    assert isinstance(y, tf.Tensor)
    assert y.shape[0] == 2
    assert y.shape[-1] == 2


def test_model_preserves_spatial_resolution(model):
    """
    For pixel-wise segmentation, the prediction should have the same
    spatial resolution as the input.
    """

    x = tf.random.normal((1, 224, 224, 3))
    y = model(x, training=False)

    assert y.shape[1:3] == (224, 224)


# ---------------------------------------------------------------------------
# Output validity
# ---------------------------------------------------------------------------

def test_model_outputs_probabilities(model):
    """
    The final Dense layer uses softmax, so class probabilities should:

    1. be within [0, 1]
    2. sum to approximately 1 across the class dimension
    """

    x = tf.random.normal((1, 224, 224, 3))
    y = model(x, training=False)

    assert tf.reduce_all(y >= 0)
    assert tf.reduce_all(y <= 1)

    probability_sum = tf.reduce_sum(y, axis=-1)

    tf.debugging.assert_near(
        probability_sum,
        tf.ones_like(probability_sum),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Batch handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_model_supports_different_batch_sizes(model, batch_size):
    """The model should not depend on a fixed batch size."""

    x = tf.random.normal((batch_size, 224, 224, 3))
    y = model(x, training=False)

    assert y.shape == (batch_size, 224, 224, 2)


# ---------------------------------------------------------------------------
# Training / inference behavior
# ---------------------------------------------------------------------------

def test_model_can_run_in_training_mode(model):
    """The model should support Keras training mode."""

    x = tf.random.normal((1, 224, 224, 3))

    y = model(x, training=True)

    assert y.shape == (1, 224, 224, 2)


def test_model_can_run_inference_mode(model):
    """The model should support Keras inference mode."""

    x = tf.random.normal((1, 224, 224, 3))

    y = model(x, training=False)

    assert y.shape == (1, 224, 224, 2)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_model_has_gradient_flow(model):
    """
    Verify that the model is differentiable and gradients can propagate
    through trainable parameters.
    """

    x = tf.random.normal((1, 224, 224, 3))
    y_true = tf.one_hot(
        tf.zeros((1, 224, 224), dtype=tf.int32),
        depth=2,
    )

    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)

        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                y_true,
                y_pred,
            )
        )

    gradients = tape.gradient(loss, model.trainable_variables)

    assert gradients is not None

    # At least some trainable parameters must receive gradients.
    valid_gradients = [
        gradient
        for gradient in gradients
        if gradient is not None
    ]

    assert len(valid_gradients) > 0

    for gradient in valid_gradients:
        assert tf.reduce_all(tf.math.is_finite(gradient))


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

def test_model_output_contains_no_nan_or_inf(model):
    """The model should produce finite predictions for finite input."""

    x = tf.random.normal((1, 224, 224, 3))
    y = model(x, training=False)

    assert tf.reduce_all(tf.math.is_finite(y))