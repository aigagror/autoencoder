import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def encode(args, img, out_dim):
    if args.encoder == 'affine':
        output = keras.Sequential([
            layers.Flatten(),
            layers.Dense(out_dim)
        ], 'affine-encoder')(img)

    elif args.encoder == 'conv':
        # First block
        output = keras.Sequential([
            layers.Conv2D(16, 1, name='first-conv-encode'),
            layers.LayerNormalization([1, 2, 3], center=False, scale=False),
            layers.LeakyReLU(0.2)
        ], 'first-encode-block')(img)

        # Hidden blocks
        hdims = [16, 32, 64, min(128, args.hdim), min(256, args.hdim),
                 min(512, args.hdim), min(512, args.hdim), min(512, args.hdim),
                 min(512, args.hdim)]
        for i in range(len(hdims) - 1):
            output = keras.Sequential([
                layers.Conv2D(hdims[i], 3, padding='same',
                              name=f'hidden-encode-block{i}-conv1'),
                layers.LayerNormalization([1,2,3], center=False, scale=False),
                layers.LeakyReLU(0.2),
                layers.Conv2D(hdims[i + 1], 3, padding='same',
                              name=f'hidden-encode-block{i}-conv2'),
                layers.LayerNormalization([1, 2, 3], center=False, scale=False),
                layers.LeakyReLU(0.2),
                layers.AveragePooling2D(),
            ], f'hidden-encode-block{i}')(output)
            if output.shape[1] == 4:
                break

        # Last block
        output = keras.Sequential([
            layers.Conv2D(args.hdim, 3, padding='same',
                          name='last-encode-block-conv1'),
            layers.LayerNormalization([1, 2, 3], center=False, scale=False),
            layers.LeakyReLU(0.2),
            layers.Conv2D(args.hdim, 4, padding='valid',
                          name='last-encode-block-conv2'),
            layers.LayerNormalization([1, 2, 3], center=False, scale=False),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dense(out_dim, name='last-encode-block-dense')
        ], 'last-encode-block')(output)

    else:
        raise Exception(f'unknown encoder network {args.encoder}')

    # Encoder output
    tf.debugging.assert_shapes([(output, tf.TensorShape([None, out_dim]))])
    return output