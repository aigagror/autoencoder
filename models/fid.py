import numpy as np
import tensorflow as tf
from numpy import cov
from numpy import iscomplexobj
from numpy import trace
from scipy.linalg import sqrtm
from tensorflow import keras
from tqdm.auto import tqdm


class FID(keras.Model):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug

        self.preprocess = keras.applications.inception_v3.preprocess_input
        self.inception = keras.applications.InceptionV3(include_top=False, pooling='avg')
        if debug:
            self.inception.summary()

    @tf.function
    def feats(self, imgs):
        tf.debugging.assert_rank(imgs, 4)

        # Resize to inception size
        imgs = tf.image.resize(imgs, [299, 299])

        # Feed through inceptionv3
        x = (imgs + 1) * 127.5
        x = self.preprocess(x)
        return self.inception(x)

    def frechet_dist(self, act1, act2):
        # assertions
        assert isinstance(act1, np.ndarray)
        assert isinstance(act2, np.ndarray)
        assert len(act1) > 0
        assert len(act2) > 0
        tf.debugging.assert_rank(act1, 2)
        tf.debugging.assert_rank(act2, 2)

        tf.debugging.assert_all_finite(act1, f'act1 not finite\n{act1}')
        tf.debugging.assert_all_finite(act2, f'act2 not finite\n{act2}')

        # mean and covariance
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

        # assertions
        tf.debugging.assert_all_finite(mu1, f'mu1 not finite\n{mu1}\n{act1}')
        tf.debugging.assert_all_finite(mu2, f'mu2 not finite\n{mu2}\n{act2}')

        tf.debugging.assert_all_finite(sigma1, f'sigma1 not finite\n{sigma1}')
        tf.debugging.assert_all_finite(sigma2, f'sigma2 not finite\n{sigma2}')

        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2)

        # calculate sqrt of product between cov
        covmean = sigma1.dot(sigma2)
        assert np.isfinite(covmean).all(), covmean
        covmean = sqrtm(covmean)

        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real

        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def fid_score(self, ds1, ds2):
        all_feats1, all_feats2 = np.zeros([0, 2048]), np.zeros([0, 2048])

        # Make sure they're RGB
        def to_rgb(imgs):
            if imgs.shape[-1] == 1:
                imgs = tf.repeat(imgs, 3, axis=-1)
            return imgs
        ds1, ds2 = ds1.map(to_rgb, tf.data.AUTOTUNE), ds2.map(to_rgb, tf.data.AUTOTUNE)

        # Distribute
        if self.debug:
            print('strategy', self.distribute_strategy)
        ds1 = self.distribute_strategy.experimental_distribute_dataset(ds1)
        ds2 = self.distribute_strategy.experimental_distribute_dataset(ds2)

        # First dataset
        for imgs in tqdm(ds1, 'feats1', leave=False):
            feats1 = self.distribute_strategy.run(self.feats, [imgs])
            feats1 = self.distribute_strategy.gather(feats1, axis=0).numpy()
            tf.debugging.assert_all_finite(feats1, f'feats not finite\n{feats1}')
            all_feats1 = np.append(all_feats1, feats1, axis=0)

        # Second dataset
        for imgs in tqdm(ds2, 'feats2', leave=False):
            feats2 = self.distribute_strategy.run(self.feats, [imgs])
            feats2 = self.distribute_strategy.gather(feats2, axis=0).numpy()
            tf.debugging.assert_all_finite(feats2, f'feats not finite\n{feats2}')
            all_feats2 = np.append(all_feats2, feats2, axis=0)

        return self.frechet_dist(all_feats1, all_feats2)
