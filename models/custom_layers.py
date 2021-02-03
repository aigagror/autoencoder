import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, losses


def hw_flatten(x):
    # Input shape x: [BATCH, HEIGHT, WIDTH, CHANNELS]
    # flat the feature volume across the tensor width and height
    x_shape = tf.shape(x)
    return tf.reshape(x, [x_shape[0], -1, x_shape[-1]])  # return [BATCH, W*H, CHANNELS]

class NormalizeImage(layers.Layer):
    def call(self, inputs, **kwargs):
        inputs -= 0.45 * 255
        inputs /= 0.225 * 255
        return inputs

class MyMSELoss(layers.Layer):
    def __init__(self):
        super().__init__()
        self.normalize = NormalizeImage()

    def call(self, inputs):
        img, recon = inputs
        img, recon = self.normalize(img), self.normalize(recon)
        mse = losses.mse(img, recon)
        mse = tf.reduce_mean(mse)
        self.add_loss(mse)
        self.add_metric(mse, 'mse')
        return recon

def make_conv2d_trans(name, sn, **kwargs):
    if sn:
        conv2d = tfa.layers.SpectralNormalization(layers.Conv2DTranspose(**kwargs), name=name)
    else:
        conv2d = layers.Conv2DTranspose(name=name, **kwargs)
    return conv2d

def make_conv2d(name, sn, **kwargs):
    if sn:
        conv2d = tfa.layers.SpectralNormalization(layers.Conv2D(**kwargs), name=name)
    else:
        conv2d = layers.Conv2D(name=name, **kwargs)
    return conv2d


def make_dense(name, sn, **kwargs):
    if sn:
        dense = tfa.layers.SpectralNormalization(layers.Dense(**kwargs), name=name)
    else:
        dense = layers.Dense(name=name, **kwargs)
    return dense


class SelfAttention(layers.Layer):
    def __init__(self, args, nfilters):
        super().__init__()

        self.f = make_conv2d('f_x', args.sn, filters=nfilters // 8, kernel_size=1)
        self.g = make_conv2d('g_x', args.sn, filters=nfilters // 8, kernel_size=1)
        self.h = make_conv2d('h_x', args.sn, filters=nfilters, kernel_size=1)

        self.gamma = self.add_weight(name='gamma', shape=[], initializer='zeros', trainable=True)

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


class StyleConv2D(layers.Layer):
    def __init__(self, args, in_c, out_c, name):
        super().__init__(name=name)
        self.in_c, self.out_c = in_c, out_c

        self.in_scale = make_conv2d('in_scale', args.sn, filters=in_c, kernel_size=1)
        self.in_bias = make_conv2d('in_bias', args.sn, filters=in_c, kernel_size=1)

        self.conv = make_conv2d('style', args.sn, filters=out_c, kernel_size=3, padding='same')

        self.norm = tfa.layers.InstanceNormalization(name='norm')

    def call(self, input):
        # latent variable z is expected to be of shape [bsz, 1, 1, zdim]
        img, z = input

        # Style std
        in_scale = self.in_scale(z)
        in_bias = self.in_bias(z)
        img = img * in_scale + in_bias

        # Convolve
        img = self.conv(img)

        # Normalize
        img = self.norm(img)

        return img


class ConstBlock(layers.Layer):
    def __init__(self, args, name):
        super().__init__(name=name)
        self.args = args

    def build(self, input_shape):
        super().build(input_shape)
        self.seed = self.add_weight('seed', shape=[1, 4, 4, self.args.zdim], trainable=True)

    def call(self, z):
        bsz = tf.shape(z)[0]
        img = tf.repeat(self.seed, bsz, axis=0)
        return img


class FirstStyleSynthBlock(layers.Layer):
    def __init__(self, args, hdim, name):
        super().__init__(name=name)
        self.style_conv = StyleConv2D(args, args.zdim, hdim, 'style-conv')
        self.noise = LearnableNoise(args, hdim, 'noise')
        self.act = layers.LeakyReLU(args.lrelu)

    def call(self, input):
        img, z = input
        img = self.style_conv((img, z))
        img = self.noise(img)
        img = self.act(img)
        return img


class HiddenStyleSynthBlock(layers.Layer):
    def __init__(self, args, in_c, out_c, name):
        super().__init__(name=name)
        self.upsampling = layers.UpSampling2D(interpolation=args.upsample)

        self.style_conv1 = StyleConv2D(args, in_c, out_c, 'style-conv1')
        self.style_conv2 = StyleConv2D(args, out_c, out_c, 'style-conv2')

        self.noise1 = LearnableNoise(args, out_c, 'noise1')
        self.noise2 = LearnableNoise(args, out_c, 'noise2')

        self.act = layers.LeakyReLU(args.lrelu)

    def call(self, input):
        img, z = input
        img = self.upsampling(img)

        img = self.style_conv1((img, z))
        img = self.noise1(img)
        img = self.act(img)

        img = self.style_conv2((img, z))
        img = self.noise2(img)
        img = self.act(img)

        return img
