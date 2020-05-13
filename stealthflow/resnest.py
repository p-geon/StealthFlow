import numpy as np
import tensorflow as tf

from .tf_layers import MyLayer

class ResNeStBlock():
    def __init__(self, radix, cardinality, bottleneck, ratio):
        """
        (3.1 Split-Attention Block)
        K: "cardinality" hyperparameter
        R: "radix" hyperparameter
        """
        self.R = radix
        self.K = cardinality
        self.b = bottleneck # ResNeXt Bottle-Neck
        self.ratio = ratio # SE-reduction factor

        # width = b * K
        # C -> bK -> b x K -> bK -> C; c'/k = b (Fig1)

        self.mylayer = MyLayer()

    def build(self, input_shape):

        self.ch = input_shape[-1]

        self.conv_1x1 = self.mylayer.conv_1x1(self.b, activation='relu')
        self.batchnorm = self.mylayer.batchnorm()

        self.group_conv_3x3 = np.empty(shape=[self.K, self.R], dtype=object)
        self.group_batchnorm = np.empty(shape=[self.K, self.R], dtype=object)
        self.arr_dense = np.empty(shape=[self.K], dtype=object)
        self.arr_batchnorm = np.empty(shape=[self.K], dtype=object)
        self.group_dense = np.empty(shape=[self.K, self.R], dtype=object)

        for k in range(self.K):
            for r in range(self.R):
                self.group_conv_3x3[k][r] = self.mylayer.conv_3x3(self.b//self.R)
                self.group_batchnorm[k][r] = self.mylayer.batchnorm()
            self.arr_dense[k] = self.mylayer.dense(self.b//(self.R*self.ratio), activation=None)
            self.arr_batchnorm[k] = self.mylayer.batchnorm()
            for r in range(self.R):
                self.group_dense[k][r] = self.mylayer.dense(self.b//self.R, activation=None)

        self.batchnorm_final = self.mylayer.batchnorm()
        self.conv_1x1_final = self.mylayer.conv_1x1(self.ch)

    def block(self, x):
        self.build(x.shape)

        path = x # input, (H, W, C)
        x = self.conv_1x1(x) # (H, W, C) -> (H, W, b)
        x = self.batchnorm(x)
        x = tf.nn.relu(x)
        x_cardinal_path = x

        list_cardinal = []
        # cardinal k
        for k in range(self.K):
            list_split = []
            for r in range(self.R):
                x = x_cardinal_path
                #ch_start, ch_end = self.ch//self.K*k, self.ch//self.K*(k+1)
                ch_start, ch_end = self.b//self.R*r, self.b//self.R*(r+1)
                x = tf.keras.layers.Lambda(lambda x: x[:, :, :, ch_start:ch_end])(x)
                x = self.group_conv_3x3[k][r](x) # (H, W, b/r) -> (H, W, b/r)
                x = self.group_batchnorm[k][r](x)
                x = tf.nn.relu(x)

                list_split.append(x)
            # (H, W, b/r) x R

            x = tf.keras.layers.Add()(list_split) # (H, W, b/r) x R -> (H, W, b/r)
            x = tf.keras.layers.GlobalAveragePooling2D()(x) # (H, W, b/r) -> (b/r, )
            x = self.arr_dense[k](x) # (b/r, ) -> (b/r/ratio', ); ratio=4 (これどこに実験設定ある？)
            x = self.arr_batchnorm[k](x)
            x = tf.nn.relu(x)

            g = x
            list_g = []
            for r in range(self.R):
                x = g
                x = self.group_dense[k][r](x) # (b/r/ratio', ) -> (b/r, )
                x = tf.keras.layers.Reshape(target_shape=(self.b//self.R, 1))(x) # (b/r, ) -> (b/r, 1); Concatenate 用に次元拡張 (Concatenate は次元を跨いだ Softmax のため)
                list_g.append(x)
            # (b/r, 1) x R
            g_arr = tf.keras.layers.Concatenate(axis=-1)(list_g) # (b/r, 1) x R -> (b/r, R)
            a = tf.nn.softmax(g_arr, axis=-1) # "radix" 方向に Softmax

            list_V = []
            for r in range(self.R):
                a_split = tf.keras.layers.Lambda(lambda x: x[:, :, r])(a) # (b/r, R) -> (b/r, ) x R
                x = tf.keras.layers.Multiply()([list_split[r], a_split]) # (H, W, b/r) x (b/r, )
                list_V.append(x)
            # (H, W, b/r) x R

            V = tf.keras.layers.Add()(list_V) # (H, W, b/r) x R -> (H, W, b)
            list_cardinal.append(V)
        # (H, W, b) x K

        x = tf.keras.layers.Concatenate(axis=-1)(list_cardinal) # (H, W, b) x K -> (H, W, bK)
        x = self.conv_1x1_final(x) # (H, W, bk) -> (H, W, C)
        x = self.batchnorm_final(x)
        x += path # (H, W, C) + (H, W, C)
        x = tf.nn.relu(x)
        return x
