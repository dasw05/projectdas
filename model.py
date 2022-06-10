

from keras.models import Model
from keras.layers.core import Dense, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Averagepooling2D

from keras.layers.convolutional import SeparableConv2D
from tensorflow.keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D, Dropout
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten
from keras.layers import DepthwiseConv2D
from tensorflow.keras.constraints import max_norm
from matplotlib import pyplot as plt


###################################################################
############## EEGnet model changed from loaded model #############
###################################################################
def EEGInception(input_time=1000, fs=128, ncha=8, filters_per_branch=8,
                 scales_time=(500, 250, 125), dropout_rate=0.25,
                 activation='elu', n_classes=2, learning_rate=0.001):


    # start the model
    input_samples = int(input_time * fs / 1000)
    scales_samples = [int(s * fs / 1000) for s in scales_time]

    # ================================ INPUT ================================= #
    input_layer = Input((input_samples, ncha, 1))

    # ========================== BLOCK 1: INCEPTION ========================== #
    b1_units = list()
    for i in range(len(scales_samples)):
        unit = Conv2D(filters=filters_per_branch,
                      kernel_size=(scales_samples[i], 1),
                      kernel_initializer='he_normal',
                      padding='same')(input_layer)
        unit = BatchNormalization()(unit)
        unit = Activation(activation)(unit)
        unit = Dropout(dropout_rate)(unit)

        unit = DepthwiseConv2D((1, ncha),
                               use_bias=False,
                               depth_multiplier=2,
                               depthwise_constraint=max_norm(1.))(unit)
        unit = BatchNormalization()(unit)
        unit = Activation(activation)(unit)
        unit = Dropout(dropout_rate)(unit)

        b1_units.append(unit)

    # Concatenation
    b1_out = keras.layers.concatenate(b1_units, axis=3)
    b1_out = AveragePooling2D((4, 1))(b1_out)

    # ========================== BLOCK 2: INCEPTION ========================== #
    b2_units = list()
    for i in range(len(scales_samples)):
        unit = Conv2D(filters=filters_per_branch,
                      kernel_size=(int(scales_samples[i]/4), 1),
                      kernel_initializer='he_normal',
                      use_bias=False,
                      padding='same')(b1_out)
        unit = BatchNormalization()(unit)
        unit = Activation(activation)(unit)
        unit = Dropout(dropout_rate)(unit)

        b2_units.append(unit)

    # Concatenate + Average pooling
    b2_out = keras.layers.concatenate(b2_units, axis=3)
    b2_out = AveragePooling2D((2, 1))(b2_out)

    # ============================ BLOCK 3: OUTPUT =========================== #
    b3_u1 = Conv2D(filters=int(filters_per_branch*len(scales_samples)/2),
                   kernel_size=(8, 1),
                   kernel_initializer='he_normal',
                   use_bias=False,
                   padding='same')(b2_out)
    b3_u1 = BatchNormalization()(b3_u1)
    b3_u1 = Activation(activation)(b3_u1)
    b3_u1 = AveragePooling2D((2, 1))(b3_u1)
    b3_u1 = Dropout(dropout_rate)(b3_u1)

    b3_u2 = Conv2D(filters=int(filters_per_branch*len(scales_samples)/4),
                   kernel_size=(4, 1),
                   kernel_initializer='he_normal',
                   use_bias=False,
                   padding='same')(b3_u1)
    b3_u2 = BatchNormalization()(b3_u2)
    b3_u2 = Activation(activation)(b3_u2)
    b3_u2 = AveragePooling2D((2, 1))(b3_u2)
    b3_out = Dropout(dropout_rate)(b3_u2)

    # Output layer
    output_layer = Flatten()(b3_out)
    output_layer = Dense(n_classes, activation='softmax')(output_layer)
    return Model(inputs=input_layer, outputs=output_layer)
