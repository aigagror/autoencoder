import tensorflow as tf
from tensorflow import keras, nn

from models import r1_penalty


class GAN(keras.Model):
    def __init__(self, args, gen, disc):
        super().__init__()
        self.bsz = args.bsz
        self.r1_weight = args.r1
        self.gen = gen
        self.disc = disc
        self.bce = nn.sigmoid_cross_entropy_with_logits

    def call(self, imgs):
        return self.gen(imgs)

    def compile(self, d_opt, g_opt):
        super().compile()
        self.d_opt = d_opt
        self.g_opt = g_opt

        self.mean_metrics = [
            keras.metrics.Mean('bce'),
            keras.metrics.Mean('r1'),
            keras.metrics.Mean('d-real'),
            keras.metrics.Mean('d-gen'),
        ]

    def disc_step(self, img):
        gen = self.gen(img)
        with tf.GradientTape() as tape:
            d_real_logits, d_gen_logits = self.disc(img), self.disc(gen)

            real_loss = self.bce(tf.ones_like(d_real_logits), d_real_logits)
            gen_loss = self.bce(tf.zeros_like(d_gen_logits), d_gen_logits)
            bce = real_loss + gen_loss
            bce = nn.compute_average_loss(bce, global_batch_size=self.bsz)

            r1 = r1_penalty(self.disc, img)
            r1 = nn.compute_average_loss(r1, global_batch_size=self.bsz)
            loss = bce + self.r1_weight * r1

        grad = tape.gradient(loss, self.disc.trainable_weights)
        self.d_opt.apply_gradients(zip(grad, self.disc.trainable_weights))

        # Discriminator probabilities
        d_real, d_gen = tf.sigmoid(d_real_logits), tf.sigmoid(d_gen_logits)
        d_real = nn.compute_average_loss(d_real, global_batch_size=self.bsz)
        d_gen = nn.compute_average_loss(d_gen, global_batch_size=self.bsz)

        return bce, r1, d_real, d_gen

    def gen_step(self, img):
        with tf.GradientTape() as tape:
            gen = self.gen(img)
            disc_gen_logits = self.disc(gen)
            loss = self.bce(tf.ones_like(disc_gen_logits), disc_gen_logits)
            loss = nn.compute_average_loss(loss, global_batch_size=self.bsz)

        g_grad = tape.gradient(loss, self.gen.trainable_weights)
        self.g_opt.apply_gradients(zip(g_grad, self.gen.trainable_weights))
        return loss

    def train_step(self, img):
        bce, r1, d_real, d_gen = self.disc_step(img)
        self.gen_step(img)

        # Assumes the listed metrics are in the right order
        num_replicas = self.distribute_strategy.num_replicas_in_sync
        for metric, val in zip(self.metrics, [bce, r1, d_real, d_gen]):
            metric.update_state(val * num_replicas)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return self.mean_metrics
