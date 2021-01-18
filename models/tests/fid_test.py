import unittest
from models import fid
import numpy as np
import tensorflow as tf


class TestFID(unittest.TestCase):
    def setUp(self) -> None:
        self.fid_model = fid.FID()

    def test_frechet_dist(self):
        a = np.random.random([10, 2048])
        b = np.random.random([10, 2048])
        self.assertAlmostEqual(self.fid_model.frechet_dist(a, a), 0, delta=1e-3)
        self.assertGreater(self.fid_model.frechet_dist(a, b), 350)

    def test_fid_score(self):
        for imsize in [32, 128, 299]:
            a = tf.data.Dataset.from_tensor_slices(255 * tf.random.uniform([10, imsize, imsize, 3])).batch(1)
            b = tf.data.Dataset.from_tensor_slices(255 * tf.random.uniform([10, imsize, imsize, 3])).batch(1)

            self.assertAlmostEqual(self.fid_model.fid_score(a, a), 0, delta=1e-3)
            self.assertGreater(self.fid_model.fid_score(a, b), 10)



if __name__ == '__main__':
    unittest.main()
