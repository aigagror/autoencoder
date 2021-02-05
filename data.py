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
            train_files = tf.data.Dataset.list_files(f'gs://aigagror/datasets/celeba_hq/{args.imsize}/train-*',
                                                     shuffle=True)
            val_data = tf.data.TFRecordDataset(f'gs://aigagror/datasets/celeba_hq/{args.imsize}/val-00000-of-00001')
        else:
            # Google Drive
            train_files = tf.data.Dataset.list_files(f'/gdrive/MyDrive/datasets/celeba_hq/train-*', shuffle=True)
            val_data = tf.data.TFRecordDataset(f'/gdrive/MyDrive/datasets/celeba_hq/val-00000-of-00001')

        train_data = train_files.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)

        def decode_rgb(data):
            img = tf.image.decode_jpeg(data, channels=3)
            img = tf.image.resize(img, [args.imsize, args.imsize])
            img = tf.cast(img, tf.uint8)
            return img

        ds_train = train_data.map(decode_rgb, tf.data.AUTOTUNE)
        ds_val = val_data.map(decode_rgb, tf.data.AUTOTUNE)

    elif args.data.endswith('mnist'):
        # MNIST
        img_c = 1
        rand_flip = False
        train_size, val_size = 50000, 10000

        def get_img(input):
            img = input['image']
            img = tf.image.resize(img, [args.imsize, args.imsize])
            img = tf.cast(img, tf.uint8)
            return img

        ds_train = tfds.load('mnist', try_gcs=True, split='train', shuffle_files=True)
        ds_val = tfds.load('mnist', try_gcs=True, split='test', shuffle_files=True)

        ds_train = ds_train.map(get_img, tf.data.AUTOTUNE)
        ds_val = ds_val.map(get_img, tf.data.AUTOTUNE)

        if args.data.startswith('fake-'):
            ds_train = ds_train.take(32)
            ds_val = ds_val.take(32)
            train_size, val_size = 32, 32

    else:
        raise Exception(f'unknown data {args.data}')

    # Cache and shuffle
    ds_train = ds_train.cache().shuffle(train_size)
    ds_val = ds_val.cache().shuffle(val_size)

    # Repeat train dataset
    ds_train = ds_train.repeat()

    # Preprocess
    def preprocess(img):
        if rand_flip:
            img = tf.image.random_flip_left_right(img)
        return img

    ds_train = ds_train.map(preprocess, tf.data.AUTOTUNE)
    ds_val = ds_val.map(preprocess, tf.data.AUTOTUNE)

    # Batch and prefetch
    ds_train = ds_train.batch(args.bsz).prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val.batch(args.bsz).prefetch(tf.data.AUTOTUNE)

    info = {
        'channels': img_c,
        'train-size': train_size,
        'val-size': val_size
    }

    return ds_train, ds_val, info
