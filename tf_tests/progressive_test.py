import unittest

import numpy as np
import tensorflow as tf

import models
import utils


class ProgressiveCheckpoints(unittest.TestCase):

    def test_progressive_save_and_load(self):
        # Pretend to train on 16x16
        args_16x16 = '--imsize=16 --model=gan --bsz=32 --train-last '
        args_16x16 = utils.parser.parse_args(args_16x16.split())
        utils.setup(args_16x16)
        model_16x16 = models.make_model(args_16x16, img_c=3, summarize=False)

        # "train"
        for w in model_16x16.weights:
            w.assign(tf.random.normal(w.shape))
        model_16x16.gen.save_weights('out/gen.h5')
        model_16x16.disc.save_weights('out/disc.h5')

        # Assert load and saving with same image size works
        args_16x16_load = '--imsize=16 --model=gan --bsz=32 --load '
        args_16x16_load = utils.parser.parse_args(args_16x16_load.split())
        model_16x16_2 = models.make_model(args_16x16_load, img_c=3, summarize=False)
        for a, b in zip(model_16x16.weights, model_16x16_2.weights):
            self.assertEqual(a.name, b.name)
            if a.name == 'total:0' or a.name == 'count:0':
                # Metric variable
                continue
            np.testing.assert_allclose(a.numpy(), b.numpy(), err_msg=a.name)

        # Now progress to 32x32 and add train last argument
        args_32x32 = '--imsize=32 --model=gan --bsz=32 --load --train-last '
        args_32x32 = utils.parser.parse_args(args_32x32.split())
        utils.setup(args_32x32)
        model_32x32 = models.make_model(args_32x32, img_c=3, summarize=False)

        for trained_weight in model_16x16.weights:
            name = trained_weight.name
            if name == 'total:0' or name == 'count:0':
                # Metric variables
                continue
            # Find the weight
            found = False
            for weight in model_32x32.weights:
                if name == weight.name:
                    np.testing.assert_allclose(trained_weight.numpy(), weight.numpy(), err_msg=name)
                    found = True
                    break
            print(f"{name} {'LOADED' if found else 'NOT LOADED'}")

        # Now progress to 32x32, but with different dimensions
        args_bad = '--imsize=32 --model=gan --bsz=32 --zdim=64 --hdim=16 --load '
        args_bad = utils.parser.parse_args(args_bad.split())
        utils.setup(args_bad)

        # Assert that we throw an error from different dimensions
        self.assertRaises(ValueError, lambda: models.make_model(args_bad, img_c=3, summarize=False))


if __name__ == '__main__':
    unittest.main()
