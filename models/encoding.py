import tensorflow as tf
from tensorflow.keras import layers

from models import layer_utils, affine, blocks


def encode(args, img, out_dim):
    batch_norm, spec_norm = 'bn' in args.encoder_norms, 'sn' in args.encoder_norms
    if args.encoder == 'affine':
        out = layers.Flatten()(img)
        out = affine.SnDense(out_dim, spec_norm=spec_norm)(out)

    elif args.encoder in blocks.preact_block_map:
        PreactBlockClass = blocks.preact_block_map[args.encoder]

        # Blocks
        layer_hdims = [16, 32, 64, 128, 256, 512, 512, 512, 512]
        layer_hdims = [min(h, args.hdim) for h in layer_hdims]
        out = img
        for hdim in layer_hdims:
            # Block layer
            out = PreactBlockClass(hdim, kernel_size=3, padding='same', leaky_relu=args.lrelu,
                                   batch_norm=batch_norm, spec_norm=spec_norm)(out)

            # Self-attention
            if out.shape[1] == 32:
                out = layer_utils.SelfAttention(hdim, spec_norm)(out)

            # Downsample
            out = layers.AveragePooling2D()(out)

            if out.shape[1] == 4:
                break

        # Last block
        out = blocks.PreactSingleConv(out_dim, kernel_size=4, leaky_relu=args.lrelu,
                                      batch_norm=batch_norm, spec_norm=spec_norm)(out)
        out = layers.Flatten()(out)
    else:
        raise Exception(f'unknown encoder network {args.encoder}')

    # Encoder output
    tf.debugging.assert_shapes([(out, tf.TensorShape([None, out_dim]))])
    return out
