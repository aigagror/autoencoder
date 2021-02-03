import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from models import custom_layers


def synthesize(args, z, img_c):
    hdims = [512, 512, 512, 512, 256, 128, 64, 32, 16]
    hdims = [min(h, args.hdim) for h in hdims]

    if args.synthesis == 'affine':
        img = custom_layers.make_dense('synth_affine', args.sn, units=args.imsize * args.imsize * img_c)(z)
        img = layers.Reshape([args.imsize, args.imsize, img_c])(img)
    elif args.synthesis.endswith('conv'):
        z = layers.Reshape([1, 1, z.shape[-1]])(z)

        # First block
        img = custom_layers.make_conv2d_trans('synth0_conv_t', args.sn, filters=args.hdim, kernel_size=4)(z)
        img = layers.BatchNormalization(name=f'synth0_norm1', scale=False)(img)
        img = layers.LeakyReLU(args.lrelu)(img)

        img = custom_layers.make_conv2d('synth0_conv', args.sn, filters=args.hdim, kernel_size=3, padding='same')(img)
        img = layers.BatchNormalization(name=f'synth0_norm2', scale=False)(img)
        img = layers.LeakyReLU(args.lrelu)(img)

        # Hidden blocks
        i = None
        for i in range(11 - int(np.log2(args.imsize)), len(hdims)):
            if args.synthesis.startswith('small-'):
                img = custom_layers.make_conv2d_trans(f'synth{i + 1}_conv_t', args.sn, filters=hdims[i],
                                                      kernel_size=4, strides=2, use_bias=False, padding='same')(img)
                img = layers.BatchNormalization(name=f'synth{i + 1}_norm', scale=False)(img)
                img = layers.LeakyReLU(args.lrelu)(img)

                if img.shape[1] == 32:
                    img = custom_layers.SelfAttention(args, hdims[i])(img)
            else:
                img = layers.UpSampling2D(interpolation=args.upsample)(img)

                img = custom_layers.make_conv2d(f'synth{i + 1}_conv1', args.sn, filters=hdims[i], kernel_size=3,
                                                use_bias=False, padding='same')(img)
                img = layers.BatchNormalization(name=f'synth{i + 1}_norm1', scale=False)(img)
                img = layers.LeakyReLU(args.lrelu)(img)

                if img.shape[1] == 32:
                    img = custom_layers.SelfAttention(args, hdims[i])(img)

                img = custom_layers.make_conv2d(f'synth{i + 1}_conv2', args.sn, filters=hdims[i], kernel_size=3,
                                                use_bias=False, padding='same')(img)
                img = layers.BatchNormalization(name=f'synth{i + 1}_norm2', scale=False)(img)
                img = layers.LeakyReLU(args.lrelu)(img)

        # To image
        img = custom_layers.make_conv2d(f'{hdims[i]}_to_img', args.sn, filters=img_c, kernel_size=1,
                                        padding='same')(img)

    elif args.synthesis == 'style':
        from models.custom_layers import ConstBlock, FirstStyleSynthBlock, HiddenStyleSynthBlock
        z = layers.Reshape([1, 1, z.shape[-1]])(z)

        # First block
        img = ConstBlock(args, 'const_block')(z)
        img = FirstStyleSynthBlock(args, hdims[0], name='synth0')(
            (img, z))

        # Hidden blocks
        i = None
        for i in range(10 - int(np.log2(args.imsize)), len(hdims) - 1):
            img = HiddenStyleSynthBlock(args, hdims[i], hdims[i + 1], name=f'synth{i + 1}')((img, z))

        # To image
        img = custom_layers.make_conv2d(f'{hdims[i + 1]}_to_img', args.sn, filters=img_c, kernel_size=1,
                                        padding='same')(img)

    else:
        raise Exception(f'unknown synthesis network: {args.synthesis}')

    # Synthesize
    tf.debugging.assert_shapes([(img, tf.TensorShape([None, args.imsize, args.imsize, img_c]))])

    # Image range
    img = layers.Activation('sigmoid')(img)
    img = img * 255
    return img
