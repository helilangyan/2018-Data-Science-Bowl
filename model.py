from keras.layers import Input, Conv2D, Dense, Conv2DTranspose, Reshape, Flatten, BatchNormalization
from keras.layers import LeakyReLU, Activation
from keras.activations import sigmoid
from keras.models import Model, Sequential
from keras.layers import Add, Concatenate
from keras.layers import MaxPooling2D

import numpy as np

from constant import target_shape, leaky_alpha
h, w, c = target_shape

# 图像大小的限定
# assert h == w
# assert np.log2(h) in [7, 8, 9, 10, 11]
# image_size = h


def resdual_block(input_tensor, filters1, filters2, kernel_size, data_format='channels_last', name='res_block'):
    conv1 = Conv2D(filters=filters1, kernel_size=(1, 1), strides=(
        1, 1), padding='same', data_format=data_format)
    y = conv1(input_tensor)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    conv2 = Conv2D(filters=filters2, kernel_size=kernel_size, strides=(
        1, 1), padding='same', data_format=data_format)
    y = conv2(y)
    y = BatchNormalization()(y)
    conv3 = Conv2D(filters=filters2, kernel_size=(1, 1), strides=(
        1, 1), padding='same', data_format=data_format)
    shortcut = conv3(input_tensor)
    result = Add()([shortcut, y])
    return result


def resnet_encoder():
    input_tensor = Input(shape=target_shape)
    block1 = resdual_block(input_tensor=input_tensor,
                           filters1=6, filters2=12, kernel_size=(3, 3))
    pool1 = MaxPooling2D((3, 3), strides=(2, 2),padding='same')(block1)  # width /2 64
    block2 = resdual_block(input_tensor=pool1, filters1=18,
                           filters2=24, kernel_size=(3, 3))
    pool2 = MaxPooling2D((3, 3), strides=(2, 2),padding='same')(block2)  # width /2 32
    block3 = resdual_block(input_tensor=pool2, filters1=36,
                           filters2=48, kernel_size=(3, 3))
    pool3 = MaxPooling2D((3, 3), strides=(2, 2),padding='same')(block3)  # width /2 16
    block4 = resdual_block(input_tensor=pool3, filters1=64,
                           filters2=72, kernel_size=(3, 3))
    pool4 = MaxPooling2D((3, 3), strides=(2, 2),padding='same')(block4)  # width /2 8

    conv_trans_1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(
        2, 2), padding='same', data_format='channels_last')(pool4)
    conv_trans_batch_1 = BatchNormalization()(conv_trans_1)  # 16 * 16 * 64
    concat1 = Concatenate()([conv_trans_batch_1, pool3])  # 64 + 48

    block5 = resdual_block(input_tensor=concat1, filters1=96,
                           filters2=64, kernel_size=(3, 3))
    conv_trans_2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(
        2, 2), padding='same', data_format='channels_last')(block5)
    conv_trans_batch_2 = BatchNormalization()(conv_trans_2)  # 32 * 32 * 64
    concat2 = Concatenate()([conv_trans_batch_2, pool2])  # 32 + 64

    block6 = resdual_block(input_tensor=concat2, filters1=48,
                           filters2=32, kernel_size=(3, 3))
    conv_trans_3 = Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(
        2, 2), padding='same', data_format='channels_last')(block6)
    conv_trans_batch_3 = BatchNormalization()(conv_trans_3)  # 64 * 64 * 16
    concat3 = Concatenate()([conv_trans_batch_3, pool1])  # 16 + 12

    block7 = resdual_block(input_tensor=concat3,
                           filters1=12, filters2=6, kernel_size=(3, 3))
    conv_trans_4 = Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=(
        2, 2), padding='same', data_format='channels_last')(block7)
    conv_trans_batch_4 = BatchNormalization()(conv_trans_4)  # 128 * 128 * 3

    conv_trans_5 = Conv2DTranspose(filters=1, kernel_size=(1, 1), strides=(
        1, 1), padding='same', data_format='channels_last')(conv_trans_batch_4)
    conv_trans_batch_5 = BatchNormalization()(conv_trans_5)  # 1

    model = Model(inputs=input_tensor,outputs=conv_trans_batch_5)
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model


def auto_encoder():
    net = Sequential()

    # ============ encode =================
    cell = Input(shape=target_shape)
    net.add(cell)
    encode_conv1 = Conv2D(filters=4, kernel_size=5, strides=(
        2, 2), padding='same', data_format='channels_last', input_shape=target_shape)
    net.add(encode_conv1)
    print(encode_conv1.output_shape)
    assert encode_conv1.output_shape[1:] == (target_shape/2, target_shape/2, 4)
    net.add(BatchNormalization())
    net.add(LeakyReLU(alpha=leaky_alpha))
    net.add(Conv2D(filters=8, kernel_size=5, strides=(2, 2),
                   padding='same', data_format='channels_last'))
    net.add(BatchNormalization())
    net.add(LeakyReLU(alpha=leaky_alpha))
    net.add(Conv2D(filters=16, kernel_size=5, strides=(2, 2),
                   padding='same', data_format='channels_last'))
    net.add(BatchNormalization())
    net.add(LeakyReLU(alpha=leaky_alpha))
    net.add(Conv2D(filters=32, kernel_size=5, strides=(2, 2),
                   padding='same', data_format='channels_last'))
    net.add(BatchNormalization())
    net.add(LeakyReLU(alpha=leaky_alpha))
    net.add(Conv2D(filters=64, kernel_size=5, strides=(2, 2),
                   padding='same', data_format='channels_last'))
    net.add(BatchNormalization())
    net.add(LeakyReLU(alpha=leaky_alpha))
    net.add(Conv2D(filters=128, kernel_size=5, strides=(
        2, 2), padding='same', data_format='channels_last'))
    net.add(BatchNormalization())
    net.add(LeakyReLU(alpha=leaky_alpha))

    # ================ decode ======================
    net.add(Conv2DTranspose(filters=64, kernel_size=5, strides=(
        2, 2), data_format='channels_last', padding='same'))
    net.add(BatchNormalization())
    net.add(LeakyReLU(alpha=leaky_alpha))
    net.add(Conv2DTranspose(filters=32, kernel_size=5, strides=(
        2, 2), data_format='channels_last', padding='same'))
    net.add(BatchNormalization())
    net.add(LeakyReLU(alpha=leaky_alpha))
    net.add(Conv2DTranspose(filters=16, kernel_size=5, strides=(
        2, 2), data_format='channels_last', padding='same'))
    net.add(BatchNormalization())
    net.add(LeakyReLU(alpha=leaky_alpha))
    net.add(Conv2DTranspose(filters=8, kernel_size=5, strides=(
        2, 2), data_format='channels_last', padding='same'))
    net.add(BatchNormalization())
    net.add(LeakyReLU(alpha=leaky_alpha))
    net.add(Conv2DTranspose(filters=4, kernel_size=5, strides=(
        2, 2), data_format='channels_last', padding='same'))
    net.add(BatchNormalization())
    net.add(LeakyReLU(alpha=leaky_alpha))
    net.add(Conv2DTranspose(filters=1, kernel_size=5, strides=(
        2, 2), data_format='channels_last', padding='same'))
    # net.add(Activation('sigmoid'))
    assert net.output_shape[1:] == target_shape
    net.compile(optimizer='adam', loss='mean_squared_error')
    return net
