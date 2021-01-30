import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers


def encode(args, img, out_dim):
    if args.encoder == 'affine':
        out = layers.Flatten()(img)
        out = layers.Dense(out_dim, name='affine-encoder')(out)

    elif args.encoder == 'conv':
        # First block
        out = tfa.layers.SpectralNormalization(layers.Conv2D(16, 1), name=f'first-encode-block-conv')(img)
        out = layers.LeakyReLU(0.1)(out)

        # Hidden blocks
        hdims = [min(16, args.hdim), min(32, args.hdim), min(64, args.hdim), min(128, args.hdim), min(256, args.hdim),
                 min(512, args.hdim), min(512, args.hdim), min(512, args.hdim), min(512, args.hdim)]
        i, in_h, out_h = None, None, None
        for i in range(len(hdims) - 1):
            prefix = f'hidden-encode-block{i}'
            in_h, out_h = hdims[i], hdims[i + 1]

            out = tfa.layers.SpectralNormalization(layers.Conv2D(in_h, 3, padding='same'),
                                                      name=f'{prefix}-conv1')(out)
            out = layers.LeakyReLU(0.1)(out)

            out = tfa.layers.SpectralNormalization(layers.Conv2D(out_h, 3, padding='same'),
                                                      name=f'{prefix}-conv2')(out)
            out = layers.LeakyReLU(0.1)(out)

            out = layers.AveragePooling2D()(out)

            if out.shape[1] == 4:
                break

        # Last block
        prefix = f'last-encode-block{i}'
        out = tfa.layers.SpectralNormalization(layers.Conv2D(out_h, 3, padding='same'),
                                                  name=f'{prefix}-conv1')(out)
        out = layers.LeakyReLU(0.1)(out)

        out = tfa.layers.SpectralNormalization(layers.Conv2D(out_h, 4, padding='valid'),
                                                  name=f'{prefix}-conv2')(out)
        out = layers.LeakyReLU(0.1)(out)

        out = layers.Flatten()(out)
        out = tfa.layers.SpectralNormalization(layers.Dense(out_dim), name=f'{prefix}-dense')(out)
    else:
        raise Exception(f'unknown encoder network {args.encoder}')

    # Encoder output
    tf.debugging.assert_shapes([(out, tf.TensorShape([None, out_dim]))])
    return out
