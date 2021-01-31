import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, losses


def hw_flatten(x):
    # Input shape x: [BATCH, HEIGHT, WIDTH, CHANNELS]
    # flat the feature volume across the tensor width and height
    x_shape = tf.shape(x)
    return tf.reshape(x, [x_shape[0], -1, x_shape[-1]])  # return [BATCH, W*H, CHANNELS]


class MyMSELoss(layers.Layer):
    def call(self, inputs):
        img, recon = inputs
        mse = losses.mse(img, recon)
        mse = tf.reduce_mean(mse)
        self.add_loss(mse)
        self.add_metric(mse, 'mse')
        return recon


class SelfAttention(layers.Layer):
    def __init__(self, number_of_filters):
        super(SelfAttention, self).__init__()

        self.f = tfa.layers.SpectralNormalization(layers.Conv2D(number_of_filters // 8, 1, padding='SAME'), name="f_x")
        self.g = tfa.layers.SpectralNormalization(layers.Conv2D(number_of_filters // 8, 1, padding='SAME'), name="g_x")
        self.h = tfa.layers.SpectralNormalization(layers.Conv2D(number_of_filters, 1, padding='SAME'), name="h_x")

        self.gamma = self.add_weight(shape=[], initializer='zeros', trainable=True)

        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        f = self.f(x)
        g = self.g(x)
        h = self.h(x)

        f_flatten = hw_flatten(f)
        g_flatten = hw_flatten(g)
        h_flatten = hw_flatten(h)

        s = tf.matmul(g_flatten, f_flatten, transpose_b=True)  # [B,N,C] * [B, N, C] = [B, N, N]

        b = tf.nn.softmax(s, axis=-1)
        o = tf.matmul(b, h_flatten)
        y = self.gamma * tf.reshape(o, tf.shape(x)) + x

        return y


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
        bsz = tf.shape(img)[0]
        z = tf.random.normal([bsz, self.dim])
        return z
