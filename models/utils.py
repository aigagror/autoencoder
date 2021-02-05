import tensorflow as tf
from tensorflow.keras import layers, losses

from models.affine import SnConv2D

class MeasureNorm(layers.Layer):
    def call(self, inputs, **kwargs):
        norms = tf.linalg.norm(inputs, axis=1)
        tf.debugging.assert_rank(norms, 1)
        mean, var = tf.nn.moments(norms, axes=[0])
        self.add_metric(mean, f'{self.name}_mean')
        self.add_metric(var, f'{self.name}_var')
        return inputs

class NormalizeImage(layers.Layer):
    def call(self, inputs, **kwargs):
        inputs -= 0.45 * 255
        inputs /= 0.225 * 255
        return inputs


class STDNorm(layers.Layer):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def call(self, x):
        var = tf.math.reduce_variance(x, self.axis, keepdims=True)
        x *= tf.math.rsqrt(var + 1e-3)
        return x


class AddMSE(layers.Layer):
    def __init__(self):
        super().__init__()
        self.normalize = NormalizeImage()

    def call(self, inputs):
        img, recon = inputs
        norm_img, norm_recon = self.normalize(img), self.normalize(recon)
        mse = losses.mse(norm_img, norm_recon)
        mse = tf.reduce_mean(mse)
        self.add_loss(mse)
        self.add_metric(mse, 'mse')
        return recon


class SelfAttention(layers.Layer):
    def __init__(self, nfilters, spec_norm):
        super().__init__()

        self.nfilters = nfilters
        self.spec_norm = spec_norm
        self.f = SnConv2D(nfilters // 8, kernel_size=1, spec_norm=spec_norm)
        self.g = SnConv2D(nfilters // 8, kernel_size=1, spec_norm=spec_norm)
        self.h = SnConv2D(nfilters, kernel_size=1, spec_norm=spec_norm)

        self.gamma = self.add_weight(name='gamma', shape=[], initializer='zeros', trainable=True)

    def build(self, input_shape):
        if self.nfilters != input_shape[-1]:
            self.scale = SnConv2D(self.nfilters, kernel_size=1, spec_norm=self.spec_norm)
        else:
            self.scale = layers.Activation('linear')

    def hw_flatten(self, x):
        # Input shape x: [BATCH, HEIGHT, WIDTH, CHANNELS]
        # flat the feature volume across the tensor width and height
        x_shape = tf.shape(x)
        return tf.reshape(x, [x_shape[0], -1, x_shape[-1]])  # return [BATCH, W*H, CHANNELS]

    def call(self, x, **kwargs):
        f = self.f(x, **kwargs)
        g = self.g(x, **kwargs)
        h = self.h(x, **kwargs)

        f_flatten = self.hw_flatten(f)
        g_flatten = self.hw_flatten(g)
        h_flatten = self.hw_flatten(h)

        s = tf.matmul(g_flatten, f_flatten, transpose_b=True)  # [B,N,C] * [B, N, C] = [B, N, N]

        b = tf.nn.softmax(s, axis=-1)
        o = tf.matmul(b, h_flatten)
        y = self.gamma * tf.reshape(o, tf.shape(x)) + self.scale(x)

        return y


class LatentMap(layers.Layer):
    def __init__(self, args):
        super().__init__()
        self.dim = args.zdim

    def call(self, img):
        bsz = tf.shape(img)[0]
        z = tf.random.normal([bsz, self.dim])
        return z
