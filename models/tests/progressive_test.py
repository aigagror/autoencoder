import unittest

import numpy as np
import tensorflow as tf

import models
import utils


class ProgressiveCheckpoints(unittest.TestCase):

    def test_progressive_save_and_load(self):
        # Pretend to train on 16x16
        args_16x16 = '--imsize=16 --model=gan --bsz=32 --zdim=32 --hdim=64 --synth=style --encoder=conv '
        args_16x16 = utils.parser.parse_args(args_16x16.split())
        utils.setup(args_16x16)
        model_16x16 = models.make_model(args_16x16, img_c=3)

        # "Pretrain"
        for w in model_16x16.weights:
            w.assign(tf.random.normal(w.shape))
        model_16x16.gen.save_weights('out/gen.h5')
        model_16x16.disc.save_weights('out/disc.h5')

        # Assert load and saving with same image size works
        model_16x16_2 = models.make_model(args_16x16, img_c=3)
        model_16x16_2.gen.load_weights('out/gen.h5', by_name=True)
        model_16x16_2.disc.load_weights('out/disc.h5', by_name=True)
        for a, b in zip(model_16x16.weights, model_16x16_2.weights):
            np.testing.assert_allclose(a.numpy(), b.numpy())

        # Now progress to 32x32
        args_32x32 = '--imsize=32 --model=gan --bsz=32 --zdim=32 --hdim=64 --synth=style --encoder=conv '
        args_32x32 = utils.parser.parse_args(args_32x32.split())
        utils.setup(args_32x32)
        model_32x32 = models.make_model(args_32x32, img_c=3)
        # Assert that we can load weights without error
        model_32x32.gen.load_weights('out/gen.h5', by_name=True)
        model_32x32.disc.load_weights('out/disc.h5', by_name=True)
        for trained_weight in model_16x16.weights:
            name = trained_weight.name
            # Find the weight
            found = False
            for weight in model_32x32.weights:
                if name == weight.name:
                    np.testing.assert_allclose(trained_weight.numpy(), weight.numpy(), err_msg=name)
                    found = True
                    break
            print(f"{name} {'LOADED' if found else 'NOT LOADED'}")

        # Now progress to 32x32, but with different dimensions
        args_bad = '--imsize=32 --model=gan --bsz=32 --zdim=64 --hdim=16 --synth=style --encoder=conv '
        args_bad = utils.parser.parse_args(args_bad.split())
        utils.setup(args_bad)
        model_bad = models.make_model(args_bad, img_c=3)
        # Assert that we throw an error from different dimensions
        self.assertRaises(ValueError, lambda: model_bad.gen.load_weights('out/gen.h5', by_name=True))
        self.assertRaises(ValueError, lambda: model_bad.disc.load_weights('out/disc.h5', by_name=True))


if __name__ == '__main__':
    unittest.main()
