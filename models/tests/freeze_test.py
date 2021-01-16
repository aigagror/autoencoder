import unittest

import utils
import models

class FreezeTest(unittest.TestCase):

    def test_last_layer_train(self):
        args = '--imsize=32 ' \
               '--model=gan --encoder=conv --synthesis=style --hdim=32 --zdim=32 ' \
               '--epochs=30 --bsz=8 --train-last ' \
               '--r1=1 '
        args = utils.parser.parse_args(args.split())
        utils.setup(args)
        gan = models.make_model(args, 3, summarize=False)
        for layer in gan.disc.layers:
            if layer.name.startswith('last'):
                self.assertTrue(layer.trainable, layer.name)
            else:
                self.assertFalse(layer.trainable, layer.name)

        for layer in gan.gen.layers:
            if layer.name.endswith('to-img'):
                self.assertTrue(layer.trainable, layer.name)
            elif layer.output_shape[1:3] == (args.imsize, args.imsize):
                self.assertTrue(layer.trainable, layer.name)
            else:
                self.assertFalse(layer.trainable, layer.name)

    def test_full_train(self):
        args = '--imsize=32 ' \
               '--model=gan --encoder=conv --synthesis=style --hdim=32 --zdim=32 ' \
               '--epochs=30 --bsz=8 ' \
               '--r1=1 '
        args = utils.parser.parse_args(args.split())
        utils.setup(args)
        gan = models.make_model(args, 3, summarize=False)
        for layer in gan.layers:
            self.assertTrue(layer.trainable, layer.name)


if __name__ == '__main__':
    unittest.main()
