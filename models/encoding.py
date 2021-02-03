import tensorflow as tf
from tensorflow.keras import layers

from models import custom_layers
import tensorflow_addons as tfa

def encode(args, img, out_dim):
    if args.encoder == 'affine':
        out = layers.Flatten()(img)
        out = custom_layers.make_dense('encode_affine', args.sn, units=out_dim)(out)

    elif args.encoder.endswith('conv'):
        # First block
        out = custom_layers.make_conv2d('encode0_conv', args.sn, filters=16, kernel_size=1, padding='same')(img)
        out = layers.LeakyReLU(args.lrelu)(out)

        # Hidden blocks
        layer_hdims = [16, 32, 64, 128, 256, 512, 512, 512, 512]
        layer_hdims = [min(h, args.hdim) for h in layer_hdims]
        i, in_h, out_h = None, None, None
        for i in range(len(layer_hdims) - 1):
            in_h, out_h = layer_hdims[i], layer_hdims[i + 1]
            if args.encoder.startswith('small-'):
                # Small layer
                out = custom_layers.make_conv2d(f'encode{i + 1}_conv', args.sn, filters=in_h, kernel_size=4, strides=2,
                                                padding='same')(out)
                out = layers.LeakyReLU(args.lrelu)(out)

                if out.shape[1] == 32:
                    out = custom_layers.SelfAttention(args, in_h)(out)
            else:
                # Standard layer
                out = custom_layers.make_conv2d(f'encode{i + 1}_conv1', args.sn, filters=in_h, kernel_size=3,
                                                padding='same')(out)
                out = layers.LeakyReLU(args.lrelu)(out)

                if out.shape[1] == 32:
                    out = custom_layers.SelfAttention(args, in_h)(out)

                out = custom_layers.make_conv2d(f'encode{i + 1}_conv2', args.sn, filters=out_h, kernel_size=3,
                                                padding='same')(out)
                out = layers.LeakyReLU(args.lrelu)(out)

                out = layers.AveragePooling2D()(out)

            if out.shape[1] == 4:
                break

        # Last block
        out = custom_layers.make_conv2d(f'encode{i + 2}_conv', args.sn, filters=out_dim, kernel_size=4)(out)
        out = layers.Flatten()(out)
    else:
        raise Exception(f'unknown encoder network {args.encoder}')

    # Encoder output
    tf.debugging.assert_shapes([(out, tf.TensorShape([None, out_dim]))])
    return out
