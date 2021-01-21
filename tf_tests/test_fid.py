import datetime
import unittest

import numpy as np
import tensorflow as tf

from models import fid


class TestFID(unittest.TestCase):
    def setUp(self) -> None:
        self.fid_model = fid.FID()

    def test_frechet_dist(self):
        a = np.random.random([10, 128])
        b = np.random.random([10, 128])

        # Assert near 0 distance for equal distributions
        self.assertAlmostEqual(self.fid_model.frechet_dist(a, a), 0, delta=1e-3)

        # Assert non-trivial distance for two independent gaussians
        self.assertGreater(self.fid_model.frechet_dist(a, b), 10)

    def test_fid_score(self):
        a = tf.data.Dataset.from_tensor_slices(255 * tf.random.uniform([10, 32, 32, 3])).batch(1)
        b = tf.data.Dataset.from_tensor_slices(255 * tf.random.uniform([10, 32, 32, 3])).batch(1)

        self.assertAlmostEqual(self.fid_model.fid_score(a, a), 0, delta=1e-2)
        self.assertGreater(self.fid_model.fid_score(a, b), 10)

    def test_cifar10_fid(self):
        self.skipTest('test takes too long')
        bsz = 128
        (train_imgs, _), (val_imgs, _) = tf.keras.datasets.cifar10.load_data()
        train_imgs = tf.data.Dataset.from_tensor_slices(train_imgs).batch(bsz).prefetch(tf.data.AUTOTUNE)
        val_imgs = tf.data.Dataset.from_tensor_slices(val_imgs).batch(bsz).prefetch(tf.data.AUTOTUNE)
        start = datetime.datetime.now()
        fid = self.fid_model.fid_score(train_imgs, val_imgs)
        end = datetime.datetime.now()
        duration = end - start
        print(f'FID: {fid:.3}. Wall time: {duration}. BSZ: {bsz}')

    def test_mnist_fid(self):
        self.skipTest('test takes too long')
        bsz = 128
        (train_imgs, _), (val_imgs, _) = tf.keras.datasets.mnist.load_data()
        train_imgs = np.expand_dims(train_imgs, -1)
        val_imgs = np.expand_dims(val_imgs, -1)

        train_imgs = tf.data.Dataset.from_tensor_slices(train_imgs).batch(bsz).prefetch(tf.data.AUTOTUNE)
        val_imgs = tf.data.Dataset.from_tensor_slices(val_imgs).batch(bsz).prefetch(tf.data.AUTOTUNE)
        start = datetime.datetime.now()
        fid = self.fid_model.fid_score(train_imgs, val_imgs)
        end = datetime.datetime.now()
        duration = end - start
        print(f'FID: {fid:.3}. Wall time: {duration}. BSZ: {bsz}')


if __name__ == '__main__':
    unittest.main()
