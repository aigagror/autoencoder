import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers


class OptionalSpectralNorm(layers.Layer):
    def __init__(self, AffineLayer, *args, **kwargs):
        sn = kwargs.pop('spec_norm')
        super().__init__()
        affine = AffineLayer(*args, **kwargs)
        if sn:
            affine = tfa.layers.SpectralNormalization(affine)
        self.affine = affine

    def call(self, inputs, **kwargs):
        return self.affine(inputs, **kwargs)


class SnConv2DTranspose(OptionalSpectralNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(layers.Conv2DTranspose, *args, **kwargs)


class SnConv2D(OptionalSpectralNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(layers.Conv2D, *args, **kwargs)


class SnDense(OptionalSpectralNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(layers.Dense, *args, **kwargs)


class StyleConv2D(layers.Layer):
    def __init__(self, args, in_c, out_c, name):
        super().__init__(name=name)
        self.in_c, self.out_c = in_c, out_c

        self.in_scale = SnConv2D(in_c, kernel_size=1, spec_norm=args.sn)
        self.in_bias = SnConv2D(in_c, kernel_size=1, spec_norm=args.sn)

        self.conv = SnConv2D(out_c, kernel_size=3, padding='same', spec_norm=args.sn)

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
