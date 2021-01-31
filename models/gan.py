import tensorflow as tf
from tensorflow import keras, nn
from tqdm.auto import tqdm

from models import r1_penalty


class GAN(keras.Model):
    def __init__(self, args, gen, disc):
        super().__init__()
        self.bsz = args.bsz
        self.r1_weight = args.r1
        self.gen = gen
        self.disc = disc
        self.bce = nn.sigmoid_cross_entropy_with_logits

        # Metrics
        self.metrics_dict = {
            'disc_loss': keras.metrics.Mean('disc_loss'),
            'gen_loss': keras.metrics.Mean('gen_loss'),
            'r1': keras.metrics.Mean('r1'),

            'real_acc': keras.metrics.Mean('real_acc'),
            'gen_acc': keras.metrics.Mean('gen_acc'),

            'd_real_logits': keras.metrics.Mean('d_real_logits'),
            'd_gen_logits': keras.metrics.Mean('d_gen_logits'),
        }

    def call(self, imgs):
        return self.gen(imgs)

    @tf.function
    def tf_gen(self, imgs):
        return self.gen(imgs)

    def gen_ds(self, ds):
        """
        Generates fake dataset the same size as the given dataset
        """
        ds = self.distribute_strategy.experimental_distribute_dataset(ds)
        all_gen_imgs = []
        for imgs in tqdm(ds, 'gen_ds'):
            gen_imgs = self.distribute_strategy.run(self.tf_gen, [imgs])
            gen_imgs = self.distribute_strategy.gather(gen_imgs, axis=0)
            all_gen_imgs.append(gen_imgs)
        ds_gen = tf.concat(all_gen_imgs, axis=0)
        tf.debugging.assert_greater_equal(ds_gen, -1.0)
        tf.debugging.assert_less_equal(ds_gen, 1.0)
        ds_gen = tf.data.Dataset.from_tensor_slices(ds_gen).batch(self.bsz).prefetch(tf.data.AUTOTUNE)
        return ds_gen

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return self.metrics_dict.values()

    def disc_hinge_loss(self, d_real_logits, d_gen_logits):
        real_loss = tf.nn.relu(1. - d_real_logits)
        gen_loss = tf.nn.relu(1. + d_gen_logits)
        return real_loss + gen_loss

    def gen_hinge_loss(self, d_gen_logits):
        return -d_gen_logits

    def disc_step(self, img):
        gen = self.gen(img, training=True)
        with tf.GradientTape() as tape:
            d_real_logits, d_gen_logits = self.disc(img, training=True), self.disc(gen, training=True)
            real_labels = tf.ones_like(d_real_logits)
            gen_labels = tf.zeros_like(d_gen_logits)

            disc_loss = self.bce(real_labels, d_real_logits) + self.bce(gen_labels, d_gen_logits)
            disc_loss = nn.compute_average_loss(disc_loss, global_batch_size=self.bsz)

            r1 = r1_penalty(self.disc, img)
            r1 = nn.compute_average_loss(r1, global_batch_size=self.bsz)
            loss = disc_loss + self.r1_weight * r1

        grad = tape.gradient(loss, self.disc.trainable_weights)
        self.disc.optimizer.apply_gradients(zip(grad, self.disc.trainable_weights))

        # Discriminator probabilities and accuracies
        real_acc = keras.metrics.binary_accuracy(real_labels, d_real_logits, threshold=0)
        gen_acc = keras.metrics.binary_accuracy(gen_labels, d_gen_logits, threshold=0)

        d_real_logits = nn.compute_average_loss(d_real_logits, global_batch_size=self.bsz)
        d_gen_logits = nn.compute_average_loss(d_gen_logits, global_batch_size=self.bsz)
        real_acc = nn.compute_average_loss(real_acc, global_batch_size=self.bsz)
        gen_acc = nn.compute_average_loss(gen_acc, global_batch_size=self.bsz)

        # Return metrics
        info = {
            'disc_loss': disc_loss, 'r1': r1,
            'real_acc': real_acc, 'gen_acc': gen_acc,
            'd_real_logits': d_real_logits, 'd_gen_logits': d_gen_logits,
        }

        return info

    def gen_step(self, img):
        with tf.GradientTape() as tape:
            gen = self.gen(img, training=True)
            d_gen_logits = self.disc(gen, training=True)
            gen_loss = -d_gen_logits
            gen_loss = nn.compute_average_loss(gen_loss, global_batch_size=self.bsz)

        grad = tape.gradient(gen_loss, self.gen.trainable_weights)
        self.gen.optimizer.apply_gradients(zip(grad, self.gen.trainable_weights))

        return {'gen_loss': gen_loss}

    def train_step(self, img):
        d_metrics = self.disc_step(img)
        g_metrics = self.gen_step(img)

        # Update metrics
        num_replicas = self.distribute_strategy.num_replicas_in_sync
        for metrics in [d_metrics, g_metrics]:
            for key, val in metrics.items():
                self.metrics_dict[key].update_state(val * num_replicas)

        return {m.name: m.result() for m in self.metrics}
