import os
import shutil

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras


def plot_sample_images(model, ds):
    imgs = next(iter(ds))
    recons = model(imgs)

    imgs = imgs[:4]
    recons = recons[:4]

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


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, args, ds_val):
        self.args = args
        self.ds_val = ds_val

    def on_train_begin(self, *args):
        imgs, recons = plot_sample_images(self.model, self.ds_val)

    def on_epoch_end(self, epoch, *args):
        imgs, recons = plot_sample_images(self.model, self.ds_val)


def train(args, model, ds_train, ds_val):
    if not args.load:
        shutil.rmtree(os.path.join(args.out, 'train'), ignore_errors=True)

    callbacks = [
        # keras.callbacks.TensorBoard(args.out, histogram_freq=1, update_freq=32),
        # keras.callbacks.ModelCheckpoint(os.path.join(args.out, 'model'),
        #                                 save_weights_only=True),
        CustomCallback(args, ds_val),
    ]
    try:
        model.fit(ds_train, batch_size=args.bsz, epochs=args.epochs,
                  steps_per_epoch=28000 // args.bsz,
                  callbacks=callbacks)
    except KeyboardInterrupt:
        print('caught keyboard interrupt. ended training.')