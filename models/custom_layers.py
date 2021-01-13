import tensorflow as tf
from tensorflow.keras import layers, losses


class MyMSELoss(layers.Layer):
    def call(self, inputs):
        img, recon = inputs
        mse = losses.mse(img, recon)
        self.add_loss(mse)
        self.add_metric(mse, 'mse')
        return recon


class FooLoss(layers.Layer):
    def call(self, img):
        loss = tf.reduce_sum(img ** 2, axis=[1, 2, 3])
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
    def __init__(self, args, channels):
        super().__init__()
        self.scale = tf.Variable(tf.zeros([1, 1, 1, channels], dtype=args.dtype),
                                 name=self.name + '/scale')

    def call(self, img):
        noise = tf.random.normal(tf.shape(img), dtype=img.dtype)
        img = img + self.scale * noise
        return img


class AddBias(layers.Layer):
    def build(self, input_shape):
        self.b = tf.Variable(tf.random.normal([1] + input_shape[1:]), name='bias')

    def call(self, input):
        bsz = len(input)
        b = tf.repeat(self.b, bsz, axis=0)
        return input + b


class LatentMap(layers.Layer):
    def __init__(self, args):
        super().__init__()
        self.dim = args.zdim

    def call(self, img):
        bsz = len(img)
        z = tf.random.normal([bsz, self.dim])
        return z
