import tensorflow as tf
from tensorflow.keras import layers, losses


class MyMSELoss(layers.Layer):
    def call(self, inputs):
        img, recon = inputs
        mse = losses.mse(img, recon)
        mse = tf.reduce_mean(mse)
        self.add_loss(mse)
        self.add_metric(mse, 'mse')
        return recon


class FooLoss(layers.Layer):
    def call(self, img):
        loss = tf.reduce_sum(img ** 2, axis=[1, 2, 3])
        loss = tf.reduce_mean(loss)
        self.add_loss(loss)
        return img


class STDNorm(layers.Layer):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def call(self, x):
        var = tf.math.reduce_variance(x, self.axis, keepdims=True)
        x *= tf.math.rsqrt(var + 1e-3)
        return x


class LearnableNoise(layers.Layer):
    def __init__(self, args, channels, name):
        super().__init__(name=name)
        self.channels = channels

    def build(self, input_shape):
        super().build(input_shape)
        self.scale = self.add_weight('scale', shape=[1, 1, 1, self.channels], trainable=True)

    def call(self, img):
        noise = tf.random.normal(tf.shape(img), dtype=img.dtype)
        img = img + self.scale * noise
        return img


class LatentMap(layers.Layer):
    def __init__(self, args):
        super().__init__()
        self.dim = args.zdim

    def call(self, img):
        bsz = len(img)
        z = tf.random.normal([bsz, self.dim])
        return z
