import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers


def encode(args, img, out_dim):
    if args.encoder == 'affine':
        out = layers.Flatten()(img)
        out = layers.Dense(out_dim, name='affine')(out)

    elif args.encoder == 'conv':
        # First block
        out = tfa.layers.SpectralNormalization(layers.Conv2D(16, 1), name='block0_conv')(img)
        out = layers.LeakyReLU(0.1)(out)

        # Hidden blocks
        hdims = [min(16, args.hdim), min(32, args.hdim), min(64, args.hdim), min(128, args.hdim), min(256, args.hdim),
                 min(512, args.hdim), min(512, args.hdim), min(512, args.hdim), min(512, args.hdim)]
        i, in_h, out_h = None, None, None
        for i in range(len(hdims) - 1):
            in_h, out_h = hdims[i], hdims[i + 1]
            out = tfa.layers.SpectralNormalization(
                layers.Conv2D(in_h, 4, 2, padding='same'), name=f'block{i + 1}_conv')(out)
            out = layers.LeakyReLU(0.1)(out)

            if out.shape[1] == 4:
                break

        # Last block
        out = tfa.layers.SpectralNormalization(layers.Conv2D(out_h, 4, padding='valid'), name=f'block{i + 2}_conv')(out)
        out = layers.LeakyReLU(0.1)(out)

        out = layers.Flatten()(out)
        out = tfa.layers.SpectralNormalization(layers.Dense(out_dim), name=f'block{i + 2}_dense')(out)
    else:
        raise Exception(f'unknown encoder network {args.encoder}')

    # Encoder output
    tf.debugging.assert_shapes([(out, tf.TensorShape([None, out_dim]))])
    return out
