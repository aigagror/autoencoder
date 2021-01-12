import tensorflow as tf
from tensorflow.keras import losses


def grad_norm_sq(model, img1, img2=None):
    # Random interpolation?
    if img2 is not None:
        epsilon = tf.random.uniform([img1.shape[0], 1, 1, 1], dtype=img1.dtype)
        x = epsilon * img1 + (1 - epsilon) * img2
    else:
        x = img1

    # Gradient w.r.t input
    with tf.GradientTape() as t:
        t.watch(x)
        d_hat = model(x)
    gradients = t.gradient(d_hat, x)

    # Square sum. Assumes input is image shape
    return tf.reduce_sum(gradients ** 2, axis=[1, 2, 3])


def r1_penalty(model, img):
    return grad_norm_sq(model, img)


def grad_penalty(model, img1, img2):
    grad_norm = tf.sqrt(grad_norm_sq(model, img1, img2))
    return (grad_norm - 1) ** 2, grad_norm


def bn_loss(target_feats, gen_feats):
    target_means = tf.reduce_mean(target_feats, axis=[1, 2])
    gen_means = tf.reduce_mean(gen_feats, axis=[1, 2])

    target_stds = tf.math.reduce_std(target_feats, axis=[1, 2])
    gen_stds = tf.math.reduce_std(gen_feats, axis=[1, 2])

    loss = losses.mse(target_means, gen_means) + losses.mse(target_stds, gen_stds)
    return loss
