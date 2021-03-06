import datetime
import os
import shutil

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras


def plot_sample_images(model, ds):
    imgs = next(iter(ds))
    imgs = imgs[:4]
    recons = model(imgs)

    ret = imgs, recons

    tf.debugging.assert_type(imgs, tf.uint8)
    recons = tf.cast(recons, tf.uint8)
    if imgs.shape[-1] == 1:
        imgs = tf.squeeze(imgs, axis=-1)
        recons = tf.squeeze(recons, axis=-1)

    f, ax = plt.subplots(2, len(imgs))
    f.set_size_inches(12, 6)
    if len(imgs) == 1:
        ax[0].imshow(imgs[0], vmin=0, vmax=1)
        ax[1].imshow(recons[0], vmin=0, vmax=1)
    else:
        for i, (img, recon) in enumerate(zip(imgs, recons)):
            ax[0, i].imshow(img)
            ax[1, i].imshow(recon)

    f.tight_layout()
    plt.show()

    return ret


class PlotImagesCallback(keras.callbacks.Callback):
    def __init__(self, args, ds_val):
        super().__init__()
        self.args = args
        self.ds_val = ds_val

    def on_train_begin(self, *args):
        imgs, recons = plot_sample_images(self.model, self.ds_val)

    def on_epoch_end(self, epoch, *args):
        imgs, recons = plot_sample_images(self.model, self.ds_val)


class GANCheckpoint(keras.callbacks.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_epoch_end(self, epoch, logs=None):
        self.model.gen.save(os.path.join(self.args.out, 'gen'))
        self.model.disc.save(os.path.join(self.args.out, 'disc'))


class FIDCallback(keras.callbacks.Callback):
    def __init__(self, args, fid_model, ds_train, ds_val):
        super().__init__()
        self.args = args
        self.fid_model = fid_model
        self.ds_train = ds_train
        self.ds_val = ds_val

    def on_train_begin(self, logs=None):
        if self.args.debug:
            now = datetime.datetime.now()
            # Take at most 10000 items from train dataset to save time
            print('calculating FID score between train and val', flush=True)
            fid = self.fid_model.fid_score(self.ds_val, self.ds_train.take(10000 // self.args.bsz))
            end = datetime.datetime.now()
            duration = end - now
            print(f'{fid:.3} FID between train and val. {duration} wall time')

    def on_epoch_end(self, epoch, logs=None):
        now = datetime.datetime.now()
        ds_gen = self.model.gen_ds(self.ds_val)
        fid = self.fid_model.fid_score(self.ds_val, ds_gen)
        end = datetime.datetime.now()
        duration = end - now
        print(f'{fid:.3} FID. {duration} wall time')


def get_callbacks(args, ds_train, ds_val, fid_model):
    callbacks = [
        PlotImagesCallback(args, ds_val),
        keras.callbacks.TensorBoard(os.path.join(args.out, 'logs'), write_graph=False, update_freq=args.update_freq,
                                    profile_batch=args.profile_batch)
    ]
    if args.model == 'autoencoder' and args.save:
        model_path = os.path.join(args.out, 'ae')
        callbacks.append(keras.callbacks.ModelCheckpoint(model_path))

    elif args.model == 'gan':
        # Custom model checkpoint
        if args.save:
            callbacks.append(GANCheckpoint(args))

        # FID
        if args.fid:
            callbacks.append(FIDCallback(args, fid_model, ds_train, ds_val))

    return callbacks


def train(args, model, ds_train, ds_val, ds_info, fid_model=None):
    # Reset data?
    if not args.load:
        if args.out.startswith('gs://'):
            os.system(f"gsutil -m rm {os.path.join(args.out, '**')}")
        else:
            shutil.rmtree(args.out)
            os.mkdir(args.out)

    # Callbacks
    callbacks = get_callbacks(args, ds_train, ds_val, fid_model)

    # Train
    steps_per_epoch = args.steps_epoch
    if steps_per_epoch is None:
        steps_per_epoch = ds_info['train-size'] // args.bsz
        print(f'steps-per-epoch not specified. setting it to train-size // bsz = {steps_per_epoch}')
    try:
        model.fit(ds_train, batch_size=args.bsz, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
                  callbacks=callbacks)
    except KeyboardInterrupt:
        print('caught keyboard interrupt. ended training.')
