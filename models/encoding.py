import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


def encode(args, img, out_dim):
    if args.encoder == 'affine':
        output = keras.Sequential([
            layers.Flatten(),
            layers.Dense(out_dim, name='dense')
        ], 'affine-encoder')(img)

    elif args.encoder == 'conv':
        # First block
        prefix = 'first-encode-block'
        output = keras.Sequential([
            tfa.layers.SpectralNormalization(layers.Conv2D(16, 1), name=f'{prefix}-conv'),
            layers.LeakyReLU(0.1)
        ], prefix)(img)

        # Hidden blocks
        hdims = [min(16, args.hdim), min(32, args.hdim), min(64, args.hdim), min(128, args.hdim), min(256, args.hdim),
                 min(512, args.hdim), min(512, args.hdim), min(512, args.hdim), min(512, args.hdim)]
        i, in_h, out_h = None, None, None
        for i in range(len(hdims) - 1):
            prefix = f'hidden-encode-block{i}'
            in_h, out_h = hdims[i], hdims[i + 1]
            output = keras.Sequential([
                tfa.layers.SpectralNormalization(layers.Conv2D(in_h, 3, padding='same'),
                                                 name=f'{prefix}-conv1'),
                layers.LeakyReLU(0.1),
                tfa.layers.SpectralNormalization(layers.Conv2D(out_h, 3, padding='same'),
                                                 name=f'{prefix}-conv2'),
                layers.LeakyReLU(0.1),
                layers.AveragePooling2D(),
            ], prefix)(output)
            if output.shape[1] == 4:
                break

        # Last block
        prefix = f'last-encode-block{i}'
        output = keras.Sequential([
            tfa.layers.SpectralNormalization(layers.Conv2D(out_h, 3, padding='same'),
                                             name=f'{prefix}-conv1'),
            layers.LeakyReLU(0.1),
            tfa.layers.SpectralNormalization(layers.Conv2D(out_h, 4, padding='valid'),
                                             name=f'{prefix}-conv2'),
            layers.LeakyReLU(0.1),
            layers.Flatten(),
            tfa.layers.SpectralNormalization(layers.Dense(out_dim), name=f'{prefix}-dense')
        ], prefix)(output)

    else:
        raise Exception(f'unknown encoder network {args.encoder}')

    # Encoder output
    tf.debugging.assert_shapes([(output, tf.TensorShape([None, out_dim]))])
    return output
