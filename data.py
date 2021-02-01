import tensorflow as tf
import tensorflow_datasets as tfds


def load_datasets(args):
    if args.data == 'celeba-hq':
        # CelebA HQ
        img_c = 3
        rand_flip = True
        train_size, val_size = 28000, 2000
        if args.tpu:
            # GCS
            train_files = tf.data.Dataset.list_files(f'gs://aigagror/datasets/celeba_hq/{args.imsize}/train-*')
            val_data = tf.data.TFRecordDataset(f'gs://aigagror/datasets/celeba_hq/{args.imsize}/val-00000-of-00001')
        else:
            # Google Drive
            train_files = tf.data.Dataset.list_files(f'/gdrive/MyDrive/datasets/celeba_hq/{args.imsize}/train-*')
            val_data = tf.data.TFRecordDataset(f'/gdrive/MyDrive/datasets/celeba_hq/{args.imsize}/val-00000-of-00001')

        train_data = train_files.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)

        def decode_rgb(data):
            img = tf.image.decode_image(data, channels=3, expand_animations=False)
            return img

        ds_train = train_data.map(decode_rgb, tf.data.AUTOTUNE)
        ds_val = val_data.map(decode_rgb, tf.data.AUTOTUNE)

    elif args.data == 'mnist':
        # MNIST
        img_c = 1
        rand_flip = False
        train_size, val_size = 50000, 10000

        def get_img(input):
            return input['image']

        ds_train = tfds.load('mnist', try_gcs=True, split='train')
        ds_val = tfds.load('mnist', try_gcs=True, split='test')

        ds_train = ds_train.map(get_img, tf.data.AUTOTUNE)
        ds_val = ds_val.map(get_img, tf.data.AUTOTUNE)

    else:
        raise Exception(f'unknown data {args.data}')

    # Preprocess and cache
    def preprocess(img):
        if rand_flip:
            img = tf.image.random_flip_left_right(img)
        img = tf.cast(img, args.dtype)
        img = img / 127.5 - 1
        return img

    ds_train = ds_train.map(preprocess, tf.data.AUTOTUNE)
    ds_val = ds_val.map(preprocess, tf.data.AUTOTUNE)

    # Batch, shuffle and prefetch
    ds_train = (
        ds_train
            .shuffle(1024)
            .batch(args.bsz, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
    )

    ds_val = (
        ds_val
            .shuffle(256)
            .batch(args.bsz, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
    )

    if args.steps_epoch is not None:
        ds_train = ds_train.repeat()
        print('steps per epoch set. repeating training dataset')

    info = {
        'channels': img_c,
        'train-size': train_size,
        'val-size': val_size
    }

    return ds_train, ds_val, info
