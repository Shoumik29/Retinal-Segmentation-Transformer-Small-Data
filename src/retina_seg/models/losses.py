"""
Responsibility
    - Define Loss functions.
"""


from typing import Callable, Tuple
import tensorflow as tf
from tensorflow.keras import backend as K
from .registry import LOSSES


def _as_factory(direct_fn: Callable) -> Callable:

    def factory(**kwargs):
        def loss_fn(y_true, y_pred):
            return direct_fn(y_true, y_pred, **kwargs)
        return loss_fn

    return factory


def IoU_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def IoU_loss(y_true, y_pred):
    return -IoU_coef(y_true, y_pred)


def jaccard_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    num_classes = K.int_shape(y_pred)[-1]
    total_jaccard_score = 0
    for index in range(num_classes):
        y_true_flatten = K.flatten(y_true[:, :, :, index])
        y_pred_flatten = K.flatten(y_pred[:, :, :, index])
        intersection = y_true_flatten * y_pred_flatten
        union = (y_true_flatten + y_pred_flatten) - intersection
        total_jaccard_score += (K.sum(intersection) + K.epsilon()) / (K.sum(union) + K.epsilon())
    return total_jaccard_score / num_classes


def jaccard_loss(y_true, y_pred):
    return 1 - jaccard_score(y_true, y_pred)


def dice_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    num_classes = K.int_shape(y_pred)[-1]
    total_dice_score = 0
    for index in range(num_classes):
        y_true_flatten = K.flatten(y_true[:, :, :, index])
        y_pred_flatten = K.flatten(y_pred[:, :, :, index])
        intersection = K.sum(y_true_flatten * y_pred_flatten)
        combined = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
        total_dice_score += (2 * intersection + K.epsilon()) / (combined + K.epsilon())
    return total_dice_score / num_classes


def dice_loss(y_true, y_pred):
    return 1 - dice_score(y_true, y_pred)


def soft_dice_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    axes = tuple(range(1, len(y_pred.shape) - 1))  # skip batch and class axes
    numerator = 2.0 * tf.math.reduce_sum(y_pred * y_true, axes)
    denominator = tf.math.reduce_sum(tf.math.square(y_pred) + tf.math.square(y_true), axes)
    return tf.math.reduce_mean((numerator + K.epsilon()) / (denominator + K.epsilon()))


def soft_dice_loss(y_true, y_pred):
    return 1 - soft_dice_score(y_true, y_pred)


def focal_dice_loss(y_true, y_pred, r: float = 0.5, gamma: float = 2.0):
    
    """Asymmetric Focal Dice Loss for multi-class segmentation."""

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

    numerator1 = (1 - y_true) * y_pred + r * y_true * (1 - y_pred)
    denominator = 2 * y_true * y_pred + (1 - y_true) * y_pred + y_true * (1 - y_pred)
    numerator2 = (1 - y_true) * y_pred + (1 - r) * (1 - y_pred) ** gamma * y_true * (1 - y_pred)

    focal_dice_loss_per_class = (numerator1 / denominator) + (numerator2 / denominator)
    return tf.reduce_mean(tf.reduce_sum(focal_dice_loss_per_class, axis=-1))


def topk_loss(y_true, y_pred, k: float = 0.2):
    
    """Averages only the top-k% highest per-pixel cross-entropy losses per image."""

    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    epsilon = 1e-7
    cross_entropy = -y_true * tf.math.log(y_pred + epsilon)

    k_pixels = tf.cast(k * tf.cast(tf.shape(cross_entropy)[1], tf.float32), tf.int32)
    topk_loss_values, _ = tf.math.top_k(cross_entropy, k=k_pixels, sorted=False)

    topk_loss_per_image = tf.reduce_mean(topk_loss_values, axis=1)
    return tf.reduce_mean(topk_loss_per_image)


def weighted_categorical_crossentropy(weight_for_0: float = 0.5, weight_for_1: float = 0.5):
    
    """
    Args:
        weight_for_0: weight for class 0 (background).
        weight_for_1: weight for class 1 (foreground/vessel).
    """

    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_true = tf.cast(y_true[..., 0:2], tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        loss = -(
            weight_for_1 * y_true * tf.math.log(y_pred)
            + weight_for_0 * (1 - y_true) * tf.math.log(1 - y_pred)
        )
        return tf.reduce_mean(loss)

    return loss_fn


def weighted_dice_loss(class_weights: Tuple[float, float] = (0.2, 0.8), smooth: float = 1e-15):
    weights = tf.constant(class_weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true[..., 0:2], tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_true_f = tf.keras.layers.Flatten()(y_true)
        y_pred_f = tf.keras.layers.Flatten()(y_pred)

        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        dice_per_class = (2.0 * intersection + smooth) / (denom + smooth)
        weighted_dice = tf.reduce_sum(weights * dice_per_class) / tf.reduce_sum(weights)
        return 1.0 - weighted_dice

    return loss_fn


def categorical_crossentropy_loss():

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true[..., 0:2], tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        return tf.reduce_mean(loss)

    return loss_fn


def joint_loss(alpha: float = 0.7, beta: float = 0.3):
    
    """
    Joint loss = alpha * weighted-dice loss + beta * categorical-crossentropy loss.

    Args:
        alpha: weight for the dice term.
        beta: weight for the cross-entropy term.
    """

    ce = categorical_crossentropy_loss()
    wd = weighted_dice_loss()

    def loss_fn(y_true, y_pred):
        return alpha * wd(y_true, y_pred) + beta * ce(y_true, y_pred)

    return loss_fn


def combined_loss(k: float = 0.2):
   
    """
    Combined loss = top-k cross-entropy + dice loss.
    """

    def loss_fn(y_true, y_pred):
        return topk_loss(y_true, y_pred, k) + dice_loss(y_true, y_pred)

    return loss_fn


# Registry -- every entry is a FACTORY (see module docstring).
LOSSES.register("categorical_crossentropy")(categorical_crossentropy_loss)
LOSSES.register("weighted_categorical_crossentropy")(weighted_categorical_crossentropy)
LOSSES.register("weighted_dice")(weighted_dice_loss)
LOSSES.register("joint")(joint_loss)
LOSSES.register("combined")(combined_loss)
LOSSES.register("iou")(_as_factory(IoU_loss))
LOSSES.register("jaccard")(_as_factory(jaccard_loss))
LOSSES.register("dice")(_as_factory(dice_loss))
LOSSES.register("soft_dice")(_as_factory(soft_dice_loss))
LOSSES.register("focal_dice")(_as_factory(focal_dice_loss))
LOSSES.register("topk")(_as_factory(topk_loss))