import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from models import affine, blocks, utils


def synthesize(args, z, img_c):
    hdims = [512, 512, 512, 512, 256, 128, 64, 32, 16]
    hdims = [min(h, args.hdim) for h in hdims]

    if args.synthesis == 'affine':
        img = affine.SnDense(args.imsize * args.imsize * img_c, spec_norm=args.sn)(z)
        img = layers.Reshape([args.imsize, args.imsize, img_c])(img)

    elif args.synthesis in blocks.preact_block_map:
        # First block
        z = layers.Reshape([1, 1, z.shape[-1]])(z)
        img = affine.SnConv2DTranspose(args.hdim, kernel_size=4, spec_norm=args.sn)(z)

        # Hidden blocks
        PreactBlockClass = blocks.preact_block_map[args.synthesis]
        for i in range(11 - int(np.log2(args.imsize)), len(hdims)):
            # Block layer
            img = PreactBlockClass(hdims[i], kernel_size=3, padding='same', spec_norm=args.sn, batch_norm=args.bn,
                                   leaky_relu=args.lrelu)(img)

            # Self-attention
            if img.shape[1] == 32:
                img = utils.SelfAttention(hdims[i], args.sn)(img)

            # Upsample
            img = layers.UpSampling2D(args.upsample)(img)

        # To image
        img = blocks.PreactSingleConv(img_c, kernel_size=1, spec_norm=args.sn,
                                      batch_norm=args.bn, leaky_relu=args.lrelu)(img)

    elif args.synthesis == 'style':
        z = layers.Reshape([1, 1, z.shape[-1]])(z)

        # First block
        img = blocks.ConstBlock(args)(z)
        img = blocks.FirstStyleSynth(args, hdims[0])((img, z))

        # Hidden blocks
        for i in range(10 - int(np.log2(args.imsize)), len(hdims) - 1):
            img = blocks.HiddenStyleSynth(args, hdims[i], hdims[i + 1])((img, z))

        # To image
        img = affine.SnConv2D(img_c, kernel_size=1, spec_norm=args.sn)(img)

    else:
        raise Exception(f'unknown synthesis network: {args.synthesis}')

    # Synthesize
    tf.debugging.assert_shapes([(img, tf.TensorShape([None, args.imsize, args.imsize, img_c]))])

    # Image range
    img = layers.Activation('sigmoid')(img)
    img = img * 255
    return img
