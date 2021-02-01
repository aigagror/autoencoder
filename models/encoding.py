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
        layer_hdims = [32, 64, 128, 256, 512, 512, 512, 512]
        layer_hdims = [min(h, args.hdim) for h in layer_hdims]
        i, hdim = None, None
        for i, hdim in enumerate(layer_hdims):
            out = tfa.layers.SpectralNormalization(
                layers.Conv2D(hdim, 3, padding='same'), name=f'block{i + 1}_conv1')(out)
            out = layers.LeakyReLU(args.lrelu)(out)
            out = tfa.layers.SpectralNormalization(
                layers.Conv2D(hdim, 3, padding='same'), name=f'block{i + 1}_conv2')(out)
            out = layers.LeakyReLU(args.lrelu)(out)
            out = layers.AveragePooling2D()(out)

            if out.shape[1] == 32:
                out = custom_layers.SelfAttention(hdim)(out)

            if out.shape[1] == 4:
                break

        # Last block
        out = tfa.layers.SpectralNormalization(layers.Conv2D(hdim, 4, padding='valid'), name=f'block{i + 2}_conv')(out)
        out = layers.LeakyReLU(args.lrelu)(out)

        out = layers.Flatten()(out)
        out = tfa.layers.SpectralNormalization(layers.Dense(out_dim), name=f'block{i + 2}_dense')(out)
    else:
        raise Exception(f'unknown encoder network {args.encoder}')

    # Encoder output
    tf.debugging.assert_shapes([(out, tf.TensorShape([None, out_dim]))])
    return out
