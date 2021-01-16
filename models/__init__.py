import os

from tensorflow import keras

from models.custom_layers import LatentMap
from models.custom_losses import r1_penalty
from models.encoding import encode
from models.gan import GAN
from models.style_content import SC_VGG19
from models.synthesis import FirstStyleSynthBlock, HiddenStyleSynthBlock, synthesize


def make_model(args, img_c, summarize=True):
    if args.model == 'autoencoder':
        # Autoencoder
        img = keras.Input((args.imsize, args.imsize, img_c), name='img-in')
        z = encode(args, img, out_dim=args.zdim)
        recon = synthesize(args, z, img_c)
        recon = SC_VGG19(args)((img, recon))

        model = keras.Model(img, recon, name='autoencoder')
        model.compile(optimizer=keras.optimizers.Adam(args.ae_lr), steps_per_execution=args.steps_exec)
        if summarize:
            model.summary()

        # load weights?
        if args.load:
            print('loaded weights')
            model.load_weights(os.path.join(args.out, 'model'))
        else:
            print('starting with new weights')

    elif args.model == 'gan':
        # Generator
        gen_in = keras.Input((args.imsize, args.imsize, img_c), name='gen-in')
        z = LatentMap(args)(gen_in)
        gen_out = synthesize(args, z, img_c)
        gen = keras.Model(gen_in, gen_out, name='generator')

        # Discriminator
        disc_in = keras.Input((args.imsize, args.imsize, img_c), name='disc-in')
        disc_out = encode(args, disc_in, out_dim=1)
        disc = keras.Model(disc_in, disc_out, name='discriminator')

        # Freeze all but the last layers of the two sub nets?
        if args.train_last:
            for layer in gen.layers + disc.layers:
                layer.trainable = layer.name.startswith('last') or layer.name.endswith('to-img')

        if summarize:
            gen.summary()
            disc.summary()

        # GAN
        model = GAN(args, gen, disc)
        model.compile(d_opt=keras.optimizers.Adam(args.disc_lr),
                      g_opt=keras.optimizers.Adam(args.gen_lr),
                      steps_per_execution=args.steps_exec)

        # load weights?
        if args.load:
            print('loaded weights')
            model.gen.load_weights(os.path.join(args.out, 'gen.h5'), by_name=True)
            model.disc.load_weights(os.path.join(args.out, 'disc.h5'), by_name=True)
        else:
            print('starting with new weights')
    else:
        raise Exception(f'unknown model {args.model}')

    return model
