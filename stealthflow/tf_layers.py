import numpy as np
import tensorflow as tf

class MyLayer():
    def __init__(self, regularizer=None):
        self.regularizer = regularizer

    def batchnorm(self):
        """
        The batch normalization (BN) parameter γ is initialized to zero in the final BN operation of each block, as has been suggested for large batch training [19].
        ↑まだ対応してない。
        """
        return tf.keras.layers.BatchNormalization(axis=-1
                                        , momentum=0.99, epsilon=0.001, center=True, scale=True
                                        , beta_initializer='zeros', gamma_initializer='ones'
                                        , moving_mean_initializer='zeros', moving_variance_initializer='ones'
                                        , beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
                                        , renorm=False, renorm_clipping=None, renorm_momentum=0.99
                                        , fused=None, trainable=True, virtual_batch_size=None, adjustment=None
                                        )

    def conv_3x3(self, ch, activation='relu'):
        return tf.keras.layers.Convolution2D(ch, activation=activation
                                        , kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1)
                                        , padding='same'
                                        , use_bias=True, bias_initializer='zeros', bias_regularizer=None, bias_constraint=None
                                        , kernel_initializer='he_normal', kernel_regularizer=self.regularizer, kernel_constraint=None # Network weights are initialized using Kaiming Initialization [24]. (5.1)
                                        )

    def conv_1x1(self, ch, activation=None):
        return tf.keras.layers.Convolution2D(ch, activation=activation
                                        , kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1)
                                        , padding='same'
                                        , use_bias=False
                                        , kernel_initializer='he_normal', kernel_regularizer=None, kernel_constraint=None # Network weights are initialized using Kaiming Initialization [24]. (5.1)
                                        )

    def dense(self, ch, activation='relu'):
        return tf.keras.layers.Dense(ch, activation=activation
                                        , use_bias=True, bias_initializer='zeros', bias_regularizer=None, bias_constraint=None
                                        , kernel_initializer='he_normal', kernel_regularizer=self.regularizer, kernel_constraint=None # Network weights are initialized using Kaiming Initialization [24]. (5.1)
                                        )

class MyBlock():
    def __init__(self, regularizer=None):
        self.mylayer = MyLayer(regularizer)

    def conv_3x3_batch_relu(self, x, ch):
        x = self.mylayer.conv_3x3(ch, activation=None)(x)
        x = self.mylayer.batchnorm()(x)
        x = tf.nn.relu(x)
        return x

    def classification_MLP(self, x, num_classes, num_units=1024, dropout=0.2):
        """
        "A drop layer is inserted before the final classification layer with dropout ratio = 0.2." (5.1 Implementation Details)
        """
        LEN_FEATURE_VECTOR = x.shape[1]*x.shape[2]*x.shape[3]
        x = tf.keras.layers.Reshape(target_shape=(LEN_FEATURE_VECTOR, ))(x)
        x = tf.keras.layers.Dropout(rate=0.2)(x)
        x = self.mylayer.dense(num_units, activation="relu")(x)
        x = tf.keras.layers.Dropout(rate=0.2)(x)
        x = self.mylayer.dense(10, activation="softmax")(x)
        return x

    def classification_GAP(self, x, num_classes):
        x = self.mylayer.conv_1x1(num_classes)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.nn.softmax(x)
        return x
