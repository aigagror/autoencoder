import unittest

import tensorflow as tf

import data
import models
import utils


class StepTests(unittest.TestCase):
    def setUp(self) -> None:
        args = '--data=mnist --imsize=32 ' \
               '--model=gan --encoder=conv --synthesis=style --hdim=32 --zdim=32 ' \
               '--epochs=30 --bsz=8 ' \
               '--r1=1 '
        self.args = utils.parser.parse_args(args.split())
        utils.setup(self.args)

    def test_gan_steps(self):
        ds_train, ds_val, ds_info = data.load_datasets(self.args)
        gan = models.make_model(self.args, ds_info['channels'], summarize=False)

        img = next(iter(ds_train))
        disc_vals = gan.disc_step(img)
        gen_vals = gan.gen_step(img)

        self.assertIsInstance(disc_vals, dict)
        self.assertIsInstance(gen_vals, dict)
        for k, v in list(disc_vals.items()) + list(gen_vals.items()):
            tf.debugging.assert_shapes([
                (v, [])
            ])

if __name__ == '__main__':
    unittest.main()
