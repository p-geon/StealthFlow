# This program is based on: example of calculating the frechet inception distance in Keras for cifar10: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from skimage.transform import resize
from scipy.linalg import sqrtm

# scaling images using skimage
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		new_image = resize(image, new_shape, 0)
		images_list.append(new_image)
	return np.asarray(images_list)

"""
TF
"""
def calc_cov(x):
    # 計算が合わないので使っていない
    mean_x = tf.reduce_mean(x, axis=1)
    mean_x = tf.expand_dims(mean_x, axis=1)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float64)
    cov_xx = vx - mx
    return cov_xx

# -------------------------------------------------------
# FID (Numpy)
# -------------------------------------------------------
class FIDNumpy:
    def __init__(self, batch_size=50, scaling=True, preprocess_input=True):
        self.model_shape = (299,299,3) # Inception V3
        self.BATCH_SIZE = batch_size

        self.model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', input_shape=self.model_shape)
        self.scaling = scaling

        self.preprocess_input = preprocess_input

    """ ------------------------
    FID: Numpy
    """
    # calculate frechet inception distance
    def feat_to_fid_numpy(self, feat1, feat2):
        mu1, sigma1 = feat1.mean(axis=0), np.cov(feat1, rowvar=False)
        mu2, sigma2 = feat2.mean(axis=0), np.cov(feat2, rowvar=False)

        ssdiff = np.sum((mu1 - mu2)**2.0)

        # covmean
        npdot = sigma1.dot(sigma2)
        covmean = sqrtm(npdot)
        if np.iscomplexobj(covmean): # check and correct imaginary numbers from sqrt
            covmean = covmean.real
        return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    def calc_fid_using_numpy(self, imgs1: '4-D Tensor', imgs2: '4-D Tensor') -> 'FID':
        print("Numpy mode")
        assert len(imgs1.shape) == 4, "imgs1 shape must be 4-D Tensor"
        assert len(imgs2.shape) == 4, "imgs2 shape must be 4-D Tensor"

        assert imgs1.max() <= 1.0, "Invalid imgs1 value, over 1.0"
        assert imgs2.max() <= 1.0, "Invalid imgs2 value, over 1.0"
        assert imgs1.min() >= 0.0, "Invalid imgs1 value, under 0.0"
        assert imgs2.min() >= 0.0, "Invalid imgs2 value, under 0.0"

        assert isinstance(imgs1[0,0,0,0], np.float32), "Type Error, imgs type must be float" # arrayのうちの一つをチェック
        assert isinstance(imgs2[0,0,0,0], np.float32), "Type Error, imgs type must be float" #

        NUM_ITER = imgs1.shape[0] // self.BATCH_SIZE
        feat1 = np.zeros(shape=[0, 2048])
        feat2 = np.zeros(shape=[0, 2048])

        if(self.preprocess_input==True):
            # range: [0.0, 1.0] -> [-1.0, 1.0]
            imgs1 = tf.keras.applications.inception_v3.preprocess_input(255.0*imgs1) # [?: 0, 255] -> [float32:-1, 1]
            imgs2 = tf.keras.applications.inception_v3.preprocess_input(255.0*imgs2) # https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/preprocess_input

        for i in range(NUM_ITER):
            if(i%10 == 0): print(f"calculating features...(numpy), batch:{i+1}/{NUM_ITER}")
            imgs1_batch = imgs1[self.BATCH_SIZE*i:self.BATCH_SIZE*(i+1)]
            imgs2_batch = imgs2[self.BATCH_SIZE*i:self.BATCH_SIZE*(i+1)]

            if(self.scaling==True):
                if(imgs1_batch.shape[-1]==3 and imgs2_batch.shape[-1]==3):
                    imgs1_batch = scale_images(imgs1_batch, self.model_shape)
                    imgs2_batch = scale_images(imgs2_batch, self.model_shape)
                elif(imgs1_batch.shape[-1]==1 and imgs2_batch.shape[-1]==1):
                    # 計算量の削減
                    imgs1_batch = scale_images(imgs1_batch, (self.model_shape[0], self.model_shape[1], 1))
                    imgs2_batch = scale_images(imgs2_batch, (self.model_shape[0], self.model_shape[1], 1))

                    imgs1_batch = np.tile(imgs1_batch, (1, 1, 1, self.model_shape[2]))
                    imgs2_batch = np.tile(imgs2_batch, (1, 1, 1, self.model_shape[2]))
                else:
                    raise TypeError(f"Invalid image shape: img1/img2 shape[-1] must be equivalent, also dimensions have 1 or 3, but imgs1:{imgs1_batch.shape[-1]}, imgs2:{imgs2_batch.shape[-1]}")
            new_feat1 = self.model.predict(imgs1_batch, batch_size=self.BATCH_SIZE)
            new_feat2 = self.model.predict(imgs2_batch, batch_size=self.BATCH_SIZE)
            feat1 = np.concatenate([feat1, new_feat1], axis=0)
            feat2 = np.concatenate([feat2, new_feat2], axis=0)
        return self.feat_to_fid_numpy(feat1, feat2)

    def __call__(self, imgs1, imgs2):
        return self.calc_fid_using_numpy(imgs1, imgs2)

""" ------------------------
FID: TensorFlow
"""
class FIDTF:#(tf.keras.layers.Layer):
    def __init__(self, batch_size, scaling=True, preprocess_input=True):
        #super(FIDLayer, self).__init__()
        self.model_shape = (299,299,3) # Inception V3
        self.BATCH_SIZE = batch_size
        print(batch_size)

        self.model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', input_shape=self.model_shape)
        self.scaling = scaling

        self.preprocess_input = preprocess_input

    def rescale_color_images_tf(self, imgs1, imgs2):
        imgs1 = tf.image.resize(imgs1, size=(self.model_shape[0], self.model_shape[1]), method="nearest")
        imgs2 = tf.image.resize(imgs2, size=(self.model_shape[0], self.model_shape[1]), method="nearest")
        return imgs1, imgs2

    def rescale_glayscale_images_tf(self, imgs1, imgs2):
        imgs1 = tf.image.resize(imgs1, size=(self.model_shape[0], self.model_shape[1]), method="nearest")
        imgs2 = tf.image.resize(imgs2, size=(self.model_shape[0], self.model_shape[1]), method="nearest")
        imgs1 = tf.tile(imgs1, multiples=[1, 1, 1, 3])
        imgs2 = tf.tile(imgs2, multiples=[1, 1, 1, 3])
        return imgs1, imgs2

    @tf.function
    def feat_to_fid_tf(self, feat1, feat2):
        # Python const.
        EPS_VAL = 1e-6
        LENGTH_FEATURE_VEC = 2048

        # calc FID
        feat1 = tf.cast(feat1, dtype=tf.float64)
        feat2 = tf.cast(feat2, dtype=tf.float64)

        mu1, sigma1 = tf.reduce_mean(feat1, axis=0), tfp.stats.covariance(feat1)
        mu2, sigma2 = tf.reduce_mean(feat2, axis=0), tfp.stats.covariance(feat2)
        ssdiff = tf.reduce_sum((mu1 - mu2)** 2.0)

        eps = tf.constant(EPS_VAL, dtype=tf.float64)
        offset = tf.eye(LENGTH_FEATURE_VEC, dtype=tf.float64) * eps
        tdot = tf.tensordot(sigma1+offset, sigma2+offset, axes=1) # こちらのほうが誤差が少ない
        #tdot = tf.tensordot(sigma1, sigma2, axes=1) + offset # 誤差が多い
        covmean = tf.linalg.sqrtm(tdot)
        covmean = tf.math.real(covmean)

        fid = ssdiff + tf.linalg.trace(sigma1 + sigma2 - 2.0 * covmean)
        return tf.cast(fid, dtype=tf.float32)

    @tf.function
    def fid_tf_loop(self, i, feat1, feat2, imgs1, imgs2):
        op = tf.cond(pred=(tf.math.mod(i, 10)==0)
                , true_fn=lambda: tf.print("calculating features...(tfp), batch:", i+1, "/", imgs1.shape[0]//tf.constant(self.BATCH_SIZE))
                , false_fn=lambda: tf.no_op())

        # バッチの切り抜き。書き方が特殊, [None, 28, 28, 1] -> [bs, 28, 28, 1]
        _imgs1 = tf.slice(input_=imgs1, begin=[self.BATCH_SIZE*i, 0, 0, 0], size=[self.BATCH_SIZE, imgs1.get_shape()[1], imgs1.get_shape()[2], imgs1.get_shape()[3]])
        _imgs2 = tf.slice(input_=imgs2, begin=[self.BATCH_SIZE*i, 0, 0 ,0], size=[self.BATCH_SIZE, imgs2.get_shape()[1], imgs2.get_shape()[2], imgs2.get_shape()[3]])

        # これは tf.cond じゃなくても一貫する (実行時に決まるので)
        if(self.scaling==True):
            if(_imgs1.shape[-1]==3 and _imgs2.shape[-1]==3): # color
                _imgs1, _imgs2 = self.rescale_color_images_tf(_imgs1, _imgs2)
            elif(_imgs1.shape[-1]==1 and _imgs2.shape[-1]==1): # grayscale
                _imgs1, _imgs2 = self.rescale_glayscale_images_tf(_imgs1, _imgs2)
            else:
                raise TypeError(f"Invalid image shape: img1/img2 shape[-1] must be equivalent, also dimensions have 1 or 3, but imgs1:{_imgs1.shape[-1]}, imgs2:{_imgs2.shape[-1]}")

        _imgs1 = tf.cast(_imgs1, dtype=tf.float32)
        _imgs2 = tf.cast(_imgs2, dtype=tf.float32)

        new_feat1 = self.model(_imgs1) # (BS, 299, 299, 3) -> (BS, 2048)
        new_feat2 = self.model(_imgs2) #

        feat1 = tf.concat([feat1, new_feat1], axis=0) # (None, 2048), (BS, 2048) -> (None+BS, 2048)
        feat2 = tf.concat([feat2, new_feat2], axis=0) #

        return (tf.add(i, 1), feat1, feat2, imgs1, imgs2, ) # ←最後の一個を空けるカンマが必要

    @tf.function
    def calc_fid_using_tfp(self, imgs1: '4-D tf-Tensor', imgs2: '4-D tf-Tensor') -> '4-D tf-Tensor':
        print("TFp mode")

        # preprocessing
        if(self.preprocess_input==True):
            # range: [0.0, 1.0] -> [-1.0, 1.0]
            imgs1 = tf.keras.applications.inception_v3.preprocess_input(255.0*imgs1) # [?: 0, 255] -> [float32:-1, 1]
            imgs2 = tf.keras.applications.inception_v3.preprocess_input(255.0*imgs2) # https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/preprocess_input

        NUM_ITER = imgs1.shape[0] // tf.constant(self.BATCH_SIZE)
        i = tf.constant(0)

        feat1 = tf.zeros(shape=(0, 2048), dtype=tf.float32) # Stackする
        feat2 = tf.zeros(shape=(0, 2048), dtype=tf.float32)

        c = lambda i, _1, _2, _3, _4: tf.less(i, NUM_ITER)
        b = lambda i, feat1, feat2, imgs1, imgs2: self.fid_tf_loop(i, feat1, feat2, imgs1, imgs2)

        i, feat1, feat2, _, _ = tf.while_loop(
                  c, b, [i, feat1, feat2, imgs1, imgs2]
                , shape_invariants=[i.get_shape(), tf.TensorShape([None, 2048]), tf.TensorShape([None, 2048]), imgs1.get_shape(), imgs2.get_shape()]
          )

        fid = self.feat_to_fid_tf(feat1, feat2) # (Datalength, 2048) x 2
        return fid
    """
    def build(self, input_shape):
        pass

    def call(self, imgs1, imgs2):
        return self.calc_fid_using_tfp(imgs1, imgs2)
    """
    def __call__(self, imgs1, imgs2):
        return self.calc_fid_using_tfp(imgs1, imgs2)

if(__name__ == "__main__"):
    # pip install -i https://test.pypi.org/simple/ StealthFlow

    import tensorflow as tf
    import numpy as np
    #from stealthflow.fid import FID

    (images1, _), (images2, _) = tf.keras.datasets.mnist.load_data()
    images1 = images1.astype(np.float32).reshape(-1, 28, 28, 1)/255.0
    images2 = images2.astype(np.float32).reshape(-1, 28, 28, 1)/255.0
    images1, images2 = images1[0:10000], images2[0:10000]

    fid_score = FIDNumpy(batch_size=50, scaling=True)(images1, images2)
    print("FID numpy: ", fid_score)

    images1 = tf.constant(images1, dtype=tf.float32)
    images2 = tf.constant(images2, dtype=tf.float32)

    images1 = tf.convert_to_tensor(images1, dtype=tf.float32)
    images2 = tf.convert_to_tensor(images2, dtype=tf.float32)

    fid_score = FIDTF(batch_size=50, scaling=True)(images1, images2)
    print("FID tensorflow: ", fid_score)
