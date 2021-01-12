import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

from models.layers import LearnableNoise


class StyleConv2D(layers.Layer):
    def __init__(self, args, in_c, out_c):
        super().__init__()
        self.in_c, self.out_c = in_c, out_c
        self.in_scale = layers.Conv2D(in_c, 1, name='in-scale-conv')
        self.in_bias = layers.Conv2D(in_c, 1, name='in-bias-conv')

        self.conv = layers.Conv2D(out_c, 3, padding='same', use_bias=False,
                                  name='style-conv')
        self.norm = tfa.layers.InstanceNormalization()

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


class FirstStyleSynthBlock(layers.Layer):
    def __init__(self, args, hdim):
        super().__init__()
        self.seed = tf.Variable(tf.random.normal([1, 4, 4, args.zdim]))

        self.style_conv = StyleConv2D(args, args.zdim, hdim)
        self.noise = LearnableNoise(args, hdim)
        self.act = layers.LeakyReLU(0.2)

    def call(self, z):
        bsz = len(z)
        img = tf.tile(self.seed, [bsz, 1, 1, 1])

        img = self.style_conv((img, z))
        img = self.noise(img)
        img = self.act(img)
        return img


class HiddenStyleSynthBlock(layers.Layer):
    def __init__(self, args, in_c, out_c):
        super().__init__()
        self.upsampling = layers.UpSampling2D(interpolation='bilinear')

        self.style_conv1 = StyleConv2D(args, in_c, out_c)
        self.style_conv2 = StyleConv2D(args, out_c, out_c)

        self.noise1 = LearnableNoise(args, out_c)
        self.noise2 = LearnableNoise(args, out_c)

        self.act = layers.LeakyReLU(0.2)

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