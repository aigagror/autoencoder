import tensorflow as tf
import tensorflow_datasets as tfds


def load_datasets(args):
    if args.data == 'celeba-hq':
        # CelebA HQ
        img_c = 3
        rand_flip = True
        if args.tpu:
            # GCS
            train_files = tf.data.Dataset.list_files('gs://aigagror/datasets/celeba_hq/train-*')
            val_data = tf.data.TFRecordDataset('gs://aigagror/datasets/celeba_hq/val-00000-of-00001')
        else:
            # Google Drive
            train_files = tf.data.Dataset.list_files('/gdrive/MyDrive/datasets/celeba_hq/train-*')
            val_data = tf.data.TFRecordDataset('/gdrive/MyDrive/datasets/celeba_hq/val-00000-of-00001')

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
        img = tf.image.resize(img, [args.imsize, args.imsize])
        if rand_flip:
            img = tf.image.random_flip_left_right(img)
        img = img / 127.5 - 1
        img = tf.cast(img, args.dtype)
        return img

    ds_train = ds_train.map(preprocess, tf.data.AUTOTUNE)
    ds_val = ds_val.map(preprocess, tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_val = ds_val.cache()

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

    return ds_train, ds_val, img_c
