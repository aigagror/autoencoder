import unittest

import main
import utils


class TestMain(unittest.TestCase):
    def test_small_gan_run(self):
        self.skipTest('too long')
        args = '--data=fake-mnist --imsize=32 ' \
               '--model=gan --hdim=32 --zdim=32 --encoder=affine --synthesis=affine ' \
               '--bsz=8 --disc-lr=4e-4 --gen-lr=1e-4 --epochs=1 '
        args = utils.parser.parse_args(args.split())
        print(args)

        main.run(args)


if __name__ == '__main__':
    unittest.main()
