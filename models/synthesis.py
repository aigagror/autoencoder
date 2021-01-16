import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers

from models.custom_layers import LearnableNoise


class StyleConv2D(layers.Layer):
    def __init__(self, args, in_c, out_c, name):
        super().__init__(name=name)
        self.in_c, self.out_c = in_c, out_c
        self.in_scale = layers.Conv2D(in_c, 1, name='in-scale')
        self.in_bias = layers.Conv2D(in_c, 1, name='in-bias')

        self.conv = tfa.layers.SpectralNormalization(layers.Conv2D(out_c, 3, padding='same', use_bias=False),
                                                     name='style')
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
        bsz = len(z)
        img = tf.repeat(self.seed, bsz, axis=0)
        return img


class FirstStyleSynthBlock(layers.Layer):
    def __init__(self, args, hdim, name):
        super().__init__(name=name)
        self.style_conv = StyleConv2D(args, args.zdim, hdim, 'style-conv')
        self.noise = LearnableNoise(args, hdim, 'noise')
        self.act = layers.LeakyReLU(0.2)

    def call(self, input):
        img, z = input
        img = self.style_conv((img, z))
        img = self.noise(img)
        img = self.act(img)
        return img


class HiddenStyleSynthBlock(layers.Layer):
    def __init__(self, args, in_c, out_c, name):
        super().__init__(name=name)
        self.upsampling = layers.UpSampling2D(interpolation='bilinear')

        self.style_conv1 = StyleConv2D(args, in_c, out_c, 'style-conv1')
        self.style_conv2 = StyleConv2D(args, out_c, out_c, 'style-conv2')

        self.noise1 = LearnableNoise(args, out_c, 'noise1')
        self.noise2 = LearnableNoise(args, out_c, 'noise2')

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


def synthesize(args, z, img_c):
    hdims = [min(512, args.hdim), min(512, args.hdim), min(512, args.hdim),
             min(512, args.hdim), min(256, args.hdim), min(128, args.hdim),
             min(64, args.hdim), min(32, args.hdim), min(16, args.hdim)]
    if args.synthesis == 'affine':
        img = keras.Sequential([
            layers.Dense(args.imsize * args.imsize * img_c, activation='tanh',
                         name='dense-synth'),
            layers.Reshape([args.imsize, args.imsize, img_c])
        ], 'affine-synth')(z)

    elif args.synthesis == 'conv':
        z = layers.Reshape([1, 1, z.shape[-1]])(z)

        # First block
        prefix = f'first-conv-synth-block'
        img = keras.Sequential([
            layers.Conv2DTranspose(args.hdim, kernel_size=4, name=f'{prefix}-convt'),
            layers.LeakyReLU(0.2)
        ], prefix)(z)

        # Hidden blocks
        i = None
        for i in range(len(hdims)):
            prefix = f'hidden-conv-synth-block{i}'
            img = keras.Sequential([
                layers.UpSampling2D(interpolation='bilinear'),

                layers.Conv2D(hdims[i], 3, padding='same', name=f'{prefix}-conv1'),
                layers.LeakyReLU(0.2),

                layers.Conv2D(hdims[i], 3, padding='same', name=f'{prefix}-conv2'),
                layers.LeakyReLU(0.2),
            ], prefix)(img)

            if img.shape[1] == args.imsize:
                break

        # To image
        img = layers.Conv2D(img_c, 1, activation='tanh', name=f'{hdims[i]}-to-img')(img)

    elif args.synthesis == 'style':
        z = layers.Reshape([1, 1, z.shape[-1]])(z)

        # First block
        img = ConstBlock(args, 'const-block')(z)
        img = FirstStyleSynthBlock(args, hdims[0], name='first-style-synth-block')(
            (img, z))

        # Hidden blocks
        i = None
        for i in range(len(hdims) - 1):
            img = HiddenStyleSynthBlock(args, hdims[i], hdims[i + 1], name=f'hidden-style-synth-block{i}')((img, z))

            if img.shape[1] == args.imsize:
                break

        # To image
        img = tfa.layers.SpectralNormalization(layers.Conv2D(img_c, 1, activation='tanh'),
                                               name=f'{hdims[i + 1]}-to-img')(img)

    else:
        raise Exception(f'unknown synthesis network: {args.synthesis}')

    # Synthesize
    tf.debugging.assert_shapes([(img, tf.TensorShape([None, args.imsize, args.imsize, img_c]))])
    return img
