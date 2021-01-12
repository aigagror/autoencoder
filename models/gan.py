import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses

from models import r1_penalty


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