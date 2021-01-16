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

    def compile(self, d_opt, g_opt, **kwargs):
        super().compile(**kwargs)
        self.d_opt = d_opt
        self.g_opt = g_opt

        self.mean_metrics = [
            keras.metrics.Mean('bce'),
            keras.metrics.Mean('r1'),
            keras.metrics.Mean('real-acc'),
            keras.metrics.Mean('gen-acc'),
        ]

        self.norm_metrics = [
            keras.metrics.Mean('d-grad-norm'),
            keras.metrics.Mean('g-grad-norm'),
        ]

    def disc_step(self, img):
        gen = self.gen(img)
        with tf.GradientTape() as tape:
            d_real_logits, d_gen_logits = self.disc(img), self.disc(gen)
            real_labels = tf.ones_like(d_real_logits)
            gen_labels = tf.zeros_like(d_gen_logits)

            noisy_real_labels = real_labels + 0.05 * tf.random.uniform(tf.shape(real_labels))
            noisy_gen_labels = gen_labels + 0.05 * tf.random.uniform(tf.shape(gen_labels))
            real_loss = self.bce(noisy_real_labels, d_real_logits)
            gen_loss = self.bce(noisy_gen_labels, d_gen_logits)
            bce = real_loss + gen_loss
            bce = nn.compute_average_loss(bce, global_batch_size=self.bsz)

            r1 = r1_penalty(self.disc, img)
            r1 = nn.compute_average_loss(r1, global_batch_size=self.bsz)
            loss = bce + self.r1_weight * r1

        grad = tape.gradient(loss, self.disc.trainable_weights)
        grad = [tf.clip_by_norm(g, 2) for g in grad]
        self.d_opt.apply_gradients(zip(grad, self.disc.trainable_weights))

        # Discriminator probabilities
        real_acc = keras.metrics.binary_accuracy(real_labels, d_real_logits)
        gen_acc = keras.metrics.binary_accuracy(gen_labels, d_gen_logits)
        real_acc = nn.compute_average_loss(real_acc, global_batch_size=self.bsz)
        gen_acc = nn.compute_average_loss(gen_acc, global_batch_size=self.bsz)

        # Measure average gradient norms
        all_grad_norms = [tf.norm(g) for g in grad]
        grad_norm = tf.reduce_mean(all_grad_norms)

        return bce, r1, real_acc, gen_acc, grad_norm

    def gen_step(self, img):
        with tf.GradientTape() as tape:
            gen = self.gen(img)
            disc_gen_logits = self.disc(gen)
            loss = self.bce(tf.ones_like(disc_gen_logits), disc_gen_logits)
            loss = nn.compute_average_loss(loss, global_batch_size=self.bsz)

        grad = tape.gradient(loss, self.gen.trainable_weights)
        grad = [tf.clip_by_norm(g, 2) for g in grad]
        self.g_opt.apply_gradients(zip(grad, self.gen.trainable_weights))

        # Measure average gradient norms
        all_grad_norms = [tf.norm(g) for g in grad]
        grad_norm = tf.reduce_mean(all_grad_norms)

        return loss, grad_norm

    def train_step(self, img):
        bce, r1, d_real, d_gen, d_grad_norm = self.disc_step(img)
        _, g_grad_norm = self.gen_step(img)

        # Assumes the listed metrics are in the right order
        num_replicas = self.distribute_strategy.num_replicas_in_sync
        for mean_metric, val in zip(self.mean_metrics, [bce, r1, d_real, d_gen]):
            mean_metric.update_state(val * num_replicas)

        for norm_metric, val in zip(self.norm_metrics, [d_grad_norm, g_grad_norm]):
            norm_metric.update_state(val)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return self.mean_metrics + self.norm_metrics
