import unittest

import tensorflow as tf

import data
import utils


class MyTestCase(unittest.TestCase):
    def test_img_format(self):
        args = '--data=mnist --imsize=32  --bsz=8 '
        self.args = utils.parser.parse_args(args.split())
        utils.setup(self.args)

        ds_train, ds_val, ds_info = data.load_datasets(self.args)
        train_sample = next(iter(ds_train))
        val_sample = next(iter(ds_val))

        for sample in [train_sample, val_sample]:
            tf.debugging.assert_type(sample, tf.uint8)
            tf.debugging.assert_type(sample, tf.uint8)

            # Probablisitic asserts
            min_val, max_val = tf.reduce_min(sample), tf.reduce_max(sample)
            tf.debugging.assert_greater(max_val, 127 * tf.ones_like(max_val))


if __name__ == '__main__':
    unittest.main()
