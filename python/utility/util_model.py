from keras.models import Model
from keras.layers import Input, Convolution2D, BatchNormalization, Dense, merge
from keras.layers import Reshape, Flatten, UpSampling2D
from keras.layers import MaxPooling2D, Activation
from keras.optimizers import Adam

def hed_keras(input_shape=(64,64,1)):
    inp = Input(shape=input_shape)

    #conv1
    x = Convolution2D(64,3,3,activation='relu', border_mode='same', name='conv1_1')(inp)
    x = Convolution2D(64,3,3,activation='linear', border_mode='same', name='conv1_2')(x)
    out1 = Convolution2D(1,1,1,activation='sigmoid', border_mode='same', name='score-dsn1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    #conv2
    x = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv2_1')(x)
    x = Convolution2D(128,3,3,activation='linear', border_mode='same', name='conv2_2')(x)
    out2 = Convolution2D(1,1,1,activation='linear', border_mode='same', name='score-dsn2')(x)
    out2 = UpSampling2D(size=(2,2), name='upsample2')(out2)
    out2 = Activation('sigmoid', name='sigmoid-dsn2')(out2)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    #conv3
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_1')(x)
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_2')(x)
    x = Convolution2D(256,3,3,activation='linear', border_mode='same', name='conv3_3')(x)
    out3 = Convolution2D(1,1,1,activation='linear', border_mode='same', name='score-dsn3')(x)
    out3 = UpSampling2D(size=(4,4), name='upsample3')(out3)
    out3 = Activation('sigmoid', name='sigmoid-dsn3')(out3)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    #conv4
    x = Convolution2D(512,3,3,activation='relu', border_mode='same', name='conv4_1')(x)
    x = Convolution2D(512,3,3,activation='relu', border_mode='same', name='conv4_2')(x)
    x = Convolution2D(512,3,3,activation='linear', border_mode='same', name='conv4_3')(x)
    out4 = Convolution2D(1,1,1,activation='linear', border_mode='same', name='score-dsn4')(x)
    out4 = UpSampling2D(size=(8,8), name='upsample4')(out4)
    out4 = Activation('sigmoid', name='sigmoid-dsn4')(out4)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    #conv5
    x = Convolution2D(512,3,3,activation='relu', border_mode='same', name='conv5_1')(x)
    x = Convolution2D(512,3,3,activation='relu', border_mode='same', name='conv5_2')(x)
    x = Convolution2D(512,3,3,activation='linear', border_mode='same', name='conv5_3')(x)
    out5 = Convolution2D(1,1,1,activation='linear', border_mode='same', name='score-dsn5')(x)
    out5 = UpSampling2D(size=(16,16), name='upsample5')(out5)
    out5 = Activation('sigmoid', name='sigmoid-dsn5')(out5)

    #Merge all outputs
    out = merge([out1,out2,out3,out4,out5], mode='concat', concat_axis=3)
    out = Convolution2D(1,1,1,activation='sigmoid', border_mode='same', name='new-score-weighting')(out)
    model = Model(inp,[out,out1,out2,out3,out4,out5])

    return model
