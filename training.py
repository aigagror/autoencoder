import os
import shutil

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras


def plot_sample_images(model, ds):
    imgs = next(iter(ds))
    recons = model(imgs)

    imgs = tf.cast(imgs[:4], tf.float32)
    recons = tf.cast(recons[:4], tf.float32)

    ret = imgs, recons

    imgs = (imgs + 1) / 2
    recons = (recons + 1) / 2
    if imgs.shape[-1] == 1:
        imgs = tf.squeeze(imgs, axis=-1)
        recons = tf.squeeze(recons, axis=-1)

    f, ax = plt.subplots(2, len(imgs))
    f.set_size_inches(15, 8)
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


class ProgressiveGANCheckpoint(keras.callbacks.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_epoch_end(self, epoch, logs=None):
        self.model.gen.save_weights(os.path.join(self.args.out, 'gen.h5'))
        self.model.disc.save_weights(os.path.join(self.args.out, 'disc.h5'))


def train(args, model, ds_train, ds_val):
    # Reset data?
    if not args.load:
        if args.out.startswith('gs://'):
            os.system(f"gsutil -m rm {os.path.join(args.out, '**')}")
        else:
            shutil.rmtree(args.out)
            os.mkdir(args.out)

    # Callbacks
    log_dir = os.path.join(args.out, 'logs')
    callbacks = [
        keras.callbacks.TensorBoard(log_dir, histogram_freq=1, update_freq=32),
        PlotImagesCallback(args, ds_val),
    ]
    if args.model == 'autoencoder':
        model_path = os.path.join(args.out, 'model')
        callbacks.append(keras.callbacks.ModelCheckpoint(model_path, save_weights_only=True))
    elif args.model == 'gan':
        if not args.tpu:
            callbacks.append(ProgressiveGANCheckpoint(args))
        else:
            print('WARNING: Cannot save h5 files in GCS')

    # Train
    try:
        model.fit(ds_train, batch_size=args.bsz, epochs=args.epochs, steps_per_epoch=args.steps_epoch,
                  callbacks=callbacks)
    except KeyboardInterrupt:
        print('caught keyboard interrupt. ended training.')
