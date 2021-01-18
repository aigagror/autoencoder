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

        # Optimizers
        self.d_opt = d_opt
        self.g_opt = g_opt

        # Metrics
        self.metrics_dict = {
            'bce': keras.metrics.Mean('bce'),
            'r1': keras.metrics.Mean('r1'),

            'real_prob': keras.metrics.Mean('real_prob'),
            'gen_prob': keras.metrics.Mean('gen_prob'),
            'real_acc': keras.metrics.Mean('real_acc'),
            'gen_acc': keras.metrics.Mean('gen_acc'),

            'd_grad_norm': keras.metrics.Mean('d_grad_norm'),
            'g_grad_norm': keras.metrics.Mean('g_grad_norm'),
        }

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return self.metrics_dict.values()

    def disc_step(self, img):
        gen = self.gen(img)
        with tf.GradientTape() as tape:
            d_real_logits, d_gen_logits = self.disc(img), self.disc(gen)
            real_labels = tf.ones_like(d_real_logits)
            gen_labels = tf.zeros_like(d_gen_logits)

            noisy_real_labels = real_labels + 0.05 * tf.random.uniform(tf.shape(real_labels), dtype=real_labels.dtype)
            noisy_gen_labels = gen_labels + 0.05 * tf.random.uniform(tf.shape(gen_labels), dtype=gen_labels.dtype)
            real_loss = self.bce(noisy_real_labels, d_real_logits)
            gen_loss = self.bce(noisy_gen_labels, d_gen_logits)
            bce = real_loss + gen_loss
            bce = nn.compute_average_loss(bce, global_batch_size=self.bsz)

            r1 = r1_penalty(self.disc, img)
            r1 = nn.compute_average_loss(r1, global_batch_size=self.bsz)
            loss = bce + self.r1_weight * r1

        grad = tape.gradient(loss, self.disc.trainable_weights)
        self.d_opt.apply_gradients(zip(grad, self.disc.trainable_weights))

        # Discriminator probabilities and accuracies
        real_prob = tf.sigmoid(d_real_logits)
        gen_prob = tf.sigmoid(d_gen_logits)
        real_acc = keras.metrics.binary_accuracy(real_labels, real_prob)
        gen_acc = keras.metrics.binary_accuracy(gen_labels, gen_prob)

        real_prob = nn.compute_average_loss(real_prob, global_batch_size=self.bsz)
        gen_prob = nn.compute_average_loss(gen_prob, global_batch_size=self.bsz)
        real_acc = nn.compute_average_loss(real_acc, global_batch_size=self.bsz)
        gen_acc = nn.compute_average_loss(gen_acc, global_batch_size=self.bsz)

        # Measure average gradient norms
        num_replicas = self.distribute_strategy.num_replicas_in_sync
        all_grad_norms = [tf.norm(g) for g in grad]
        grad_norm = tf.reduce_mean(all_grad_norms) / num_replicas

        # Return metrics
        info = {
            'bce': bce, 'r1': r1,
            'real_prob': real_prob, 'gen_prob': gen_prob,
            'real_acc': real_acc, 'gen_acc': gen_acc,
            'd_grad_norm': grad_norm
        }

        return info

    def gen_step(self, img):
        with tf.GradientTape() as tape:
            gen = self.gen(img)
            disc_gen_logits = self.disc(gen)
            loss = self.bce(tf.ones_like(disc_gen_logits), disc_gen_logits)
            loss = nn.compute_average_loss(loss, global_batch_size=self.bsz)

        grad = tape.gradient(loss, self.gen.trainable_weights)
        self.g_opt.apply_gradients(zip(grad, self.gen.trainable_weights))

        # Measure average gradient norms
        num_replicas = self.distribute_strategy.num_replicas_in_sync
        all_grad_norms = [tf.norm(g) for g in grad]
        grad_norm = tf.reduce_mean(all_grad_norms) / num_replicas

        return {'g_grad_norm': grad_norm}

    def train_step(self, img):
        d_metrics = self.disc_step(img)
        g_metrics = self.gen_step(img)

        # Update metrics
        num_replicas = self.distribute_strategy.num_replicas_in_sync
        for metrics in [d_metrics, g_metrics]:
            for key, val in metrics.items():
                self.metrics_dict[key].update_state(val * num_replicas)

        return {m.name: m.result() for m in self.metrics}
