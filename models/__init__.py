import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses

from models.custom_layers import LatentMap
from models.custom_losses import r1_penalty
from models.sc import SC_VGG19
from models.style_synth import FirstStyleSynthBlock, HiddenStyleSynthBlock


def encode(args, img, out_dim):
    if args.encoder == 'affine':
        output = keras.Sequential([
            layers.Flatten(),
            layers.Dense(out_dim)
        ], 'affine-encoder')(img)

    elif args.encoder == 'conv':
        # First block
        output = keras.Sequential([
            layers.Conv2D(16, 1, name='first-conv-encode'),
            layers.LeakyReLU(0.2)
        ], 'first-encode-block')(img)

        # Hidden blocks
        hdims = [16, 32, 64, min(128, args.hdim), min(256, args.hdim),
                 min(512, args.hdim), min(512, args.hdim), min(512, args.hdim),
                 min(512, args.hdim)]
        for i in range(len(hdims) - 1):
            output = keras.Sequential([
                layers.Conv2D(hdims[i], 3, padding='same',
                              name=f'hidden-encode-block{i}-conv1'),
                layers.LeakyReLU(0.2),
                layers.Conv2D(hdims[i + 1], 3, padding='same',
                              name=f'hidden-encode-block{i}-conv2'),
                layers.LeakyReLU(0.2),
                layers.AveragePooling2D(),
            ], f'hidden-encode-block{i}')(output)
            if output.shape[1] == 4:
                break

        # Last block
        output = keras.Sequential([
            layers.Conv2D(args.hdim, 3, padding='same',
                          name='last-encode-block-conv1'),
            layers.LeakyReLU(0.2),
            layers.Conv2D(args.hdim, 4, padding='valid',
                          name='last-encode-block-conv2'),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dense(out_dim, name='last-encode-block-dense')
        ], 'last-encode-block')(output)

    else:
        raise Exception(f'unknown encoder network {args.encoder}')

    # Encoder output
    tf.debugging.assert_shapes([(output, tf.TensorShape([None, out_dim]))])
    return output


def synthesize(args, z, img_c):
    hdims = [min(512, args.hdim), min(512, args.hdim), min(512, args.hdim),
             min(512, args.hdim), min(256, args.hdim), min(128, args.hdim),
             64, 32, 16]
    start_hidx = len(hdims) - int(np.log2(args.imsize)) + 2
    if args.synthesis == 'affine':
        img = keras.Sequential([
            layers.Dense(args.imsize * args.imsize * img_c, activation='tanh',
                         name='dense-synth'),
            layers.Reshape([args.imsize, args.imsize, img_c])
        ], 'affine-synth')(z)

    elif args.synthesis == 'conv':
        z = layers.Reshape([1, 1, z.shape[-1]])(z)

        # First block
        img = keras.Sequential([
            layers.Conv2DTranspose(args.hdim, kernel_size=4,
                                   name='first-conv-synth'),
            layers.LeakyReLU(0.2)
        ], 'first-synth-block')(z)

        # Hidden blocks
        for i in range(start_hidx, len(hdims)):
            img = keras.Sequential([
                layers.UpSampling2D(interpolation='bilinear'),

                layers.Conv2D(hdims[i], 3, padding='same',
                              name=f'hidden-synth-block{i}-conv1'),
                layers.LeakyReLU(0.2),

                layers.Conv2D(hdims[i], 3, padding='same',
                              name=f'hidden-synth-block{i}-conv2'),
                layers.LeakyReLU(0.2),
            ], f'hidden-synth-block{i}')(img)

        # To image
        img = layers.Conv2D(img_c, 1, activation='tanh', name='to-img')(img)

    elif args.synthesis == 'style':
        z = layers.Reshape([1, 1, z.shape[-1]])(z)

        # First block
        img = FirstStyleSynthBlock(args, hdims[start_hidx - 1], name='first-synth-block')(z)

        # Hidden blocks
        for i in range(start_hidx - 1, len(hdims) - 1):
            img = HiddenStyleSynthBlock(args, hdims[i], hdims[i + 1], name=f'hidden-synth-block{i}')((img, z))

        # To image
        img = layers.Conv2D(img_c, 1, activation='tanh', name='to-img')(img)

    else:
        raise Exception(f'unknown synthesis network: {args.synthesis}')

    # Synthesize
    tf.debugging.assert_shapes([(img, tf.TensorShape([None, args.imsize, args.imsize, img_c]))])
    return img


class GAN(keras.Model):
    def __init__(self, args, gen, disc):
        super().__init__()
        self.r1_weight = args.r1
        self.gen = gen
        self.disc = disc
        self.bce = losses.BinaryCrossentropy(from_logits=True)

    def call(self, imgs):
        return self.gen(imgs)

    def compile(self, d_opt, g_opt):
        super().compile()
        self.d_opt = d_opt
        self.g_opt = g_opt

    def disc_step(self, img):
        gen = self.gen(img)
        with tf.GradientTape() as tape:
            d_real_logits, d_gen_logits = self.disc(img), self.disc(gen)

            real_loss = self.bce(tf.ones_like(d_real_logits), d_real_logits)
            gen_loss = self.bce(tf.zeros_like(d_gen_logits), d_gen_logits)
            bce = real_loss + gen_loss

            r1 = r1_penalty(self.disc, img)
            r1 = tf.reduce_mean(r1)
            loss = bce + self.r1_weight * r1

        grad = tape.gradient(loss, self.disc.trainable_weights)
        self.d_opt.apply_gradients(zip(grad, self.disc.trainable_weights))

        # Discriminator probabilities
        d_real, d_gen = tf.sigmoid(d_real_logits), tf.sigmoid(d_gen_logits)
        d_real = tf.reduce_mean(d_real)
        d_gen = tf.reduce_mean(d_gen)

        return bce, r1, d_real, d_gen

    def gen_step(self, img):
        with tf.GradientTape() as tape:
            gen = self.gen(img)
            disc_gen_logits = self.disc(gen)
            loss = self.bce(tf.ones_like(disc_gen_logits), disc_gen_logits)

        g_grad = tape.gradient(loss, self.gen.trainable_weights)
        self.g_opt.apply_gradients(zip(g_grad, self.gen.trainable_weights))

    def train_step(self, img):
        bce, r1, d_real, d_gen = self.disc_step(img)
        self.gen_step(img)

        return {'bce': bce, 'r1': r1, 'd-real': d_real, 'd-gen': d_gen}


def make_model(args, img_c):
    if args.model == 'autoencoder':
        # Autoencoder
        img = keras.Input((args.imsize, args.imsize, img_c), name='img-in')
        z = encode(args, img, out_dim=args.zdim)
        recon = synthesize(args, z, img_c)
        recon = SC_VGG19(args)((img, recon))

        model = keras.Model(img, recon, name='autoencoder')
        model.compile(optimizer=keras.optimizers.Adam(args.ae_lr))
        model.summary()
    elif args.model == 'gan':
        # Generator
        gen_in = keras.Input((args.imsize, args.imsize, img_c), name='gen-in')
        z = LatentMap(args)(gen_in)
        gen_out = synthesize(args, z, img_c)
        gen = keras.Model(gen_in, gen_out, name='generator')
        gen.summary()

        # Discriminator
        disc_in = keras.Input((args.imsize, args.imsize, img_c), name='disc-in')
        disc_out = encode(args, disc_in, out_dim=1)
        disc = keras.Model(disc_in, disc_out, name='discriminator')
        disc.summary()

        # GAN
        model = GAN(args, gen, disc)
        model.build([None, args.imsize, args.imsize, img_c])
        model.compile(d_opt=keras.optimizers.Adam(args.disc_lr),
                      g_opt=keras.optimizers.Adam(args.gen_lr))
    else:
        raise Exception(f'unknown model {args.model}')

    if args.load:
        print('loaded weights')
        model.load_weights(os.path.join(args.out, 'model'))
    else:
        print('starting with new weights')

    return model
