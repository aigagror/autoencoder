import unittest

import utils
import models

class FreezeTest(unittest.TestCase):

    def test_last_layer_train(self):
        args = '--imsize=32 ' \
               '--model=gan --encoder=conv --synthesis=style --hdim=32 --zdim=32 ' \
               '--epochs=30 --bsz=8 --train-last ' \
               '--r1=1 '
        self.args = utils.parser.parse_args(args.split())
        utils.setup(self.args)
        gan = models.make_model(self.args, 3, summarize=False)
        for layer in gan.gen.layers + gan.disc.layers:
            if layer.name.startswith('last') or layer.name.endswith('to-img'):
                self.assertTrue(layer.trainable, layer.name)
            else:
                self.assertFalse(layer.trainable, layer.name)

    def test_full_train(self):
        args = '--imsize=32 ' \
               '--model=gan --encoder=conv --synthesis=style --hdim=32 --zdim=32 ' \
               '--epochs=30 --bsz=8 ' \
               '--r1=1 '
        self.args = utils.parser.parse_args(args.split())
        utils.setup(self.args)
        gan = models.make_model(self.args, 3, summarize=False)
        for layer in gan.layers:
            self.assertTrue(layer.trainable, layer.name)


if __name__ == '__main__':
    unittest.main()
