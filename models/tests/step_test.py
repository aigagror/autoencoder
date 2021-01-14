import unittest

import tensorflow as tf

import data
import models
import utils


class StepTests(unittest.TestCase):
    def setUp(self) -> None:
        args = '--data=mnist --imsize=32 ' \
               '--model=gan --encoder=conv --synthesis=style --hdim=32 --zdim=32 ' \
               '--epochs=30 --bsz=128 ' \
               '--r1=1 '
        self.args = utils.parser.parse_args(args.split())
        utils.setup(self.args)

    def test_gan_steps(self):
        ds_train, ds_val, img_c = data.load_datasets(self.args)
        gan = models.make_model(self.args, img_c, summarize=False)

        img = next(iter(ds_train))
        vals = gan.disc_step(img)
        for v in vals:
            tf.debugging.assert_shapes([
                (v, [])
            ])
        gen_loss = gan.gen_step(img)
        tf.debugging.assert_shapes([
            (gen_loss, [])
        ])

if __name__ == '__main__':
    unittest.main()
