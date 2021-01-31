import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from models import custom_layers


def encode(args, img, out_dim):
    if args.encoder == 'affine':
        out = layers.Flatten()(img)
        out = layers.Dense(out_dim, name='affine')(out)

    elif args.encoder == 'conv':
        # First block
        out = tfa.layers.SpectralNormalization(layers.Conv2D(16, 1), name='block0_conv')(img)
        out = layers.LeakyReLU(args.lrelu)(out)

        # Hidden blocks
        hdims = [16, 32, 64, 128, 256, 512, 512, 512, 512]
        hdims = [min(h, args.hdim) for h in hdims]
        i, in_h, out_h = None, None, None
        for i in range(len(hdims) - 1):
            in_h, out_h = hdims[i], hdims[i + 1]
            out = tfa.layers.SpectralNormalization(
                layers.Conv2D(in_h, 4, 2, padding='same'), name=f'block{i + 1}_conv')(out)
            out = layers.LeakyReLU(args.lrelu)(out)
            if out.shape[1] == 32:
                out = custom_layers.SelfAttention(in_h)(out)

            if out.shape[1] == 4:
                break

        # Last block
        out = tfa.layers.SpectralNormalization(layers.Conv2D(out_h, 4, padding='valid'), name=f'block{i + 2}_conv')(out)
        out = layers.LeakyReLU(args.lrelu)(out)

        out = layers.Flatten()(out)
        out = tfa.layers.SpectralNormalization(layers.Dense(out_dim), name=f'block{i + 2}_dense')(out)
    else:
        raise Exception(f'unknown encoder network {args.encoder}')

    # Encoder output
    tf.debugging.assert_shapes([(out, tf.TensorShape([None, out_dim]))])
    return out
