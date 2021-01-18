import numpy as np
import tensorflow as tf
from numpy import cov
from numpy import iscomplexobj
from numpy import trace
from scipy.linalg import sqrtm
from tensorflow import keras


class FID(keras.Model):
    def __init__(self):
        super().__init__()

        self.preprocess = keras.applications.inception_v3.preprocess_input
        self.inception = keras.applications.InceptionV3(include_top=False, pooling='avg')

    def feats(self, imgs):
        x = tf.image.resize(imgs, [299, 299])
        x = self.preprocess(x)
        return self.inception(x)

    def frechet_dist(self, act1, act2):
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2)

        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))

        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real

        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def fid_score(self, ds1, ds2):
        all_feats1, all_feats2 = np.zeros([0, 2048]), np.zeros([0, 2048])

        # First dataset
        for imgs in ds1:
            feats1 = self.feats(imgs)
            all_feats1 = np.append(all_feats1, feats1, axis=0)

        # Second dataset
        for imgs in ds2:
            feats2 = self.feats(imgs)
            all_feats2 = np.append(all_feats2, feats2, axis=0)

        return self.frechet_dist(all_feats1, all_feats2)
