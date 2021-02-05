import tensorflow as tf
from tensorflow.keras import layers

from models.affine import StyleConv2D, LearnableNoise, SnConv2D


class ConstBlock(layers.Layer):
    def __init__(self, args, name=None):
        super().__init__(name=name)
        self.args = args

    def build(self, input_shape):
        super().build(input_shape)
        self.seed = self.add_weight('seed', shape=[1, 4, 4, self.args.zdim], trainable=True)

    def call(self, z):
        bsz = tf.shape(z)[0]
        img = tf.repeat(self.seed, bsz, axis=0)
        return img


class FirstStyleSynth(layers.Layer):
    def __init__(self, args, hdim, name=None):
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


class HiddenStyleSynth(layers.Layer):
    def __init__(self, args, in_c, out_c, name=None):
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


class PreActivation(layers.Layer):
    def __init__(self, batch_norm, leaky_relu):
        super().__init__()
        act = []
        if batch_norm:
            act.append(layers.BatchNormalization(scale=False))
        act.append(layers.LeakyReLU(leaky_relu))
        self.batch_norm = batch_norm
        self.leaky_relu = leaky_relu
        self.act1 = tf.keras.Sequential(act)

    def make_second_act(self):
        act = []
        if self.batch_norm:
            act.append(layers.BatchNormalization(scale=False))
        act.append(layers.LeakyReLU(self.leaky_relu))
        self.act2 = tf.keras.Sequential(act)


class PreactSingleConv(PreActivation):
    def __init__(self, filters, **kwargs):
        batch_norm, lrelu = kwargs.pop('batch_norm'), kwargs.pop('leaky_relu')
        super().__init__(batch_norm, lrelu)

        self.conv = SnConv2D(filters, **kwargs)

    def call(self, inputs, **kwargs):
        x = self.act1(inputs)
        x = self.conv(x)
        return x


class PreactDoubleConv(PreActivation):
    def __init__(self, filters, **kwargs):
        batch_norm, lrelu = kwargs.pop('batch_norm'), kwargs.pop('leaky_relu')
        super().__init__(batch_norm, lrelu)
        self.make_second_act()
        self.conv1 = SnConv2D(filters, **kwargs)
        self.conv2 = SnConv2D(filters, **kwargs)

    def call(self, inputs, **kwargs):
        x = self.act1(inputs)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x


class PreactResidual(PreActivation):
    def __init__(self, filters, **kwargs):
        batch_norm, lrelu = kwargs.pop('batch_norm'), kwargs.pop('leaky_relu')
        super().__init__(batch_norm, lrelu)
        self.make_second_act()
        self.filters = filters
        self.sn = kwargs['spec_norm']
        self.conv1 = SnConv2D(filters, **kwargs)
        self.conv2 = SnConv2D(filters, **kwargs)

    def build(self, input_shape):
        if self.filters != input_shape[-1]:
            self.scale = SnConv2D(self.filters, kernel_size=1, spec_norm=self.sn)
        else:
            self.scale = layers.Activation('linear')

    def call(self, inputs, **kwargs):
        x0 = self.scale(inputs)
        x = self.act1(inputs)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x0 + x


preact_block_map = {
    'conv-1': PreactSingleConv,
    'conv-2': PreactDoubleConv,
    'resnet': PreactResidual
}
