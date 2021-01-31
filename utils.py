import argparse

import tensorflow as tf
from tensorflow.keras import mixed_precision

parser = argparse.ArgumentParser()

# Data
parser.add_argument('--data', type=str)
parser.add_argument('--imsize', type=int)

# Model
parser.add_argument('--model', choices=['autoencoder', 'gan'])
parser.add_argument('--lrelu', type=float, default=0.1)
parser.add_argument('--zdim', type=int, default=512)
parser.add_argument('--hdim', type=int, default=512)
parser.add_argument('--encoder', choices=['affine', 'conv'], default='conv')
parser.add_argument('--synthesis', choices=['affine', 'conv', 'style'], default='style')
parser.add_argument('--r1', type=float, default=0)

# Style Content Model
parser.add_argument('--style-layer', type=int)
parser.add_argument('--content-layer', type=int)
parser.add_argument('--alpha', type=float)
parser.add_argument('--train-last', action='store_true')

# Training
parser.add_argument('--epochs', type=int)
parser.add_argument('--ae-lr', type=float, default=1e-3)
parser.add_argument('--gen-lr', type=float, default=1e-3)
parser.add_argument('--disc-lr', type=float, default=4e-3)
parser.add_argument('--bsz', type=int)
parser.add_argument('--fid', action='store_true')

parser.add_argument('--steps-exec', type=int, help='steps per execution')
parser.add_argument('--steps-epoch', type=int, help='steps per epoch')
parser.add_argument('--update-freq', type=str, default='epoch', help='tensorboard update frequency')

# Save
parser.add_argument('--out', type=str, default='./out')
parser.add_argument('--load', action='store_true')

# Strategy
parser.add_argument('--tpu', action='store_true')
parser.add_argument('--policy', choices=['mixed_bfloat16', 'float32'], default='float32')

# Other
parser.add_argument('--debug', action='store_true')


def setup(args):
    # Logging
    tf.get_logger().setLevel('DEBUG' if args.debug else 'WARNING')

    # Policy
    mixed_precision.set_global_policy(args.policy)
    for d in ['bfloat16', 'float16', 'float32']:
        if d in args.policy:
            args.dtype = d
            break

    # Device and strategy
    if args.tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    else:
        strategy = tf.distribute.get_strategy()
    return strategy
