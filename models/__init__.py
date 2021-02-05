import os

from tensorflow import keras

from models.affine import SnConv2D
from models.custom_losses import r1_penalty
from models.encoding import encode
from models.gan import GAN
from models.layer_utils import NormalizeImage, AddMSE, LatentMap, MeasureNorm
from models.style_content import SC_VGG19
from models.synthesis import synthesize


def make_model(args, img_c):
    if args.model == 'autoencoder':
        if args.load:
            model = keras.models.load_model(os.path.join(args.out, 'ae'))
            print('Loaded model')
        else:
            # Autoencoder
            img = keras.Input((args.imsize, args.imsize, img_c), name='img-in')
            out = NormalizeImage()(img)
            out = encode(args, out, out_dim=args.zdim)
            out = MeasureNorm(name='latent_norm')(out)
            out = synthesize(args, out, img_c)

            out = AddMSE()((img, out))

            model = keras.Model(img, out, name='autoencoder')
            model.compile(optimizer=keras.optimizers.Adam(args.ae_lr, args.beta1), steps_per_execution=args.steps_exec)

            print('Starting with new model')

        # Summarize
        model.summary()

    elif args.model == 'gan':
        # Generator and discriminator
        if args.load:
            # Loading model
            gen = keras.models.load_model(os.path.join(args.out, 'gen'))
            disc = keras.models.load_model(os.path.join(args.out, 'disc'))
            print('Loaded model')
        else:
            # Generator
            gen_in = keras.Input((args.imsize, args.imsize, img_c), name='gen-in')
            z = LatentMap(args)(gen_in)
            gen_out = synthesize(args, z, img_c)
            gen = keras.Model(gen_in, gen_out, name='generator')

            # Discriminator
            disc_in = keras.Input((args.imsize, args.imsize, img_c), name='disc-in')
            disc_out = NormalizeImage()(disc_in)
            disc_out = encode(args, disc_out, out_dim=1)
            disc = keras.Model(disc_in, disc_out, name='discriminator')

            gen.compile(keras.optimizers.Adam(args.gen_lr, args.beta1))
            disc.compile(keras.optimizers.Adam(args.disc_lr, args.beta1))

            print('Starting with new model')

        # Summarize
        disc.summary()
        gen.summary()

        # GAN
        model = GAN(args, gen, disc)
        model.compile(steps_per_execution=args.steps_exec)

    else:
        raise Exception(f'unknown model {args.model}')

    return model
