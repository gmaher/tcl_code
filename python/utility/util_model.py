from keras.models import Model
from keras.layers import Input, Convolution2D, BatchNormalization, Dense, merge
from keras.layers import Reshape, Flatten, UpSampling2D
from keras.layers import MaxPooling2D, Activation
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np
def FCN(input_shape=(64,64,1), Nfilters=32, Wfilter=3,num_conv_1=3, num_conv_2=3,
output_channels=1, mask=True, dense_layers=1,dense_size=64, obg=False, l2_reg=0.0):
    '''
    Makes an FCN neural network
    '''
    x = Input(shape=input_shape)

    #main branch
    d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu',
    border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    d = BatchNormalization()(d)

    for i in range(0,num_conv_1):
        d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu',
        border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)
        d = BatchNormalization()(d)

    d = Convolution2D(1,Wfilter,Wfilter,activation='relu', border_mode='same',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

    #mask layer
    if mask:
    	m = Flatten()(x)

    	for i in range(0,dense_layers):
    		m = Dense(dense_size, activation='relu',
            W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    	m = Dense(input_shape[0]*input_shape[1], activation='relu',
        W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    	m = Reshape(input_shape, name="mask")(m)

    	#merge
    	d = merge([d,m], mode='mul')

    #finetune
    for i in range(0,num_conv_2):
        d = BatchNormalization()(d)
        d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu',
        border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

    d = BatchNormalization()(d)
    dout = Convolution2D(output_channels,
    Wfilter,Wfilter,activation='linear', border_mode='same',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

    if obg:
        d_softmax_input = Reshape((input_shape[0]*input_shape[1],output_channels))(dout)
        d_out = Activation('softmax', name='softmax')(d_softmax_input)
        FCN_categorical = Model(x,d_out)

        d = Reshape((input_shape[0],input_shape[1],output_channels))(d_out)
        FCN = Model(x,d)
        return FCN,FCN_categorical
    else:
        dout = Activation('sigmoid', name='sigmoid')(dout)
        FCN = Model(x,dout)
        FCN_f = Model(x,d)
        return FCN, FCN_f

def FCN_multi(FCN_f,input_shape=(64,64,1), Nfilters=32, Wfilter=3, output_channels=1, l2_reg=0):
    x = Input(shape=input_shape)

    fcn_out = FCN_f(x)

    d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(fcn_out)
    d = BatchNormalization()(d)
    d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)
    d = BatchNormalization()(d)
    d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)
    d = BatchNormalization()(d)
    # d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same',
    # W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)
    # d = BatchNormalization()(d)
    d = Convolution2D(output_channels,Wfilter,Wfilter,activation='sigmoid', border_mode='same',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

    #merge
    FCN_multi = Model(x,d)
    return FCN_multi

def OBG_FCN(FCN,OBP_FCN,input_shape=(64,64,1), Nfilters=32, Wfilter=3, output_channels=1, l2_reg=0):
    x = Input(shape=input_shape)

    fcn_out = FCN(x)

    obp_out = OBP_FCN(x)

    d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(obp_out)
    d = BatchNormalization()(d)
    d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)
    d = BatchNormalization()(d)
    d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)
    d = BatchNormalization()(d)
    d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)
    d = BatchNormalization()(d)
    d = Convolution2D(output_channels,Wfilter,Wfilter,activation='sigmoid', border_mode='same',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

    #merge
    d = merge([d,fcn_out], mode='mul')
    OBG_FCN = Model(x,d)
    return OBG_FCN

def hed_keras(input_shape=(64,64,1), Wfilter=3, Nfilters=32, dense_layers=1, dense_size=64,
num_conv=2, mask=True, l2_reg=0):
    inp = Input(shape=input_shape)

    #conv1
    x = Convolution2D(64,3,3,activation='relu', border_mode='same', name='conv1_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(inp)
    x = BatchNormalization()(x)
    x = Convolution2D(64,3,3,activation='linear', border_mode='same', name='conv1_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    out1 = Convolution2D(1,1,1,activation='sigmoid', border_mode='same', name='score-dsn1',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    #conv2
    x = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv2_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(128,3,3,activation='linear', border_mode='same', name='conv2_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    out2 = Convolution2D(1,1,1,activation='linear', border_mode='same', name='score-dsn2',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    out2 = UpSampling2D(size=(2,2), name='upsample2')(out2)
    out2 = Activation('sigmoid', name='sigmoid-dsn2')(out2)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    #conv3
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256,3,3,activation='linear', border_mode='same', name='conv3_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    out3 = Convolution2D(1,1,1,activation='linear', border_mode='same', name='score-dsn3',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    out3 = UpSampling2D(size=(4,4), name='upsample3')(out3)
    out3 = Activation('sigmoid', name='sigmoid-dsn3')(out3)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
    #
    # #conv4
    # x = Convolution2D(512,3,3,activation='relu', border_mode='same', name='conv4_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    # x = Convolution2D(512,3,3,activation='relu', border_mode='same', name='conv4_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    # x = Convolution2D(512,3,3,activation='linear', border_mode='same', name='conv4_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    # out4 = Convolution2D(1,1,1,activation='linear', border_mode='same', name='score-dsn4', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    # out4 = UpSampling2D(size=(8,8), name='upsample4')(out4)
    # out4 = Activation('sigmoid', name='sigmoid-dsn4')(out4)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)
    #
    # #conv5
    # x = Convolution2D(512,3,3,activation='relu', border_mode='same', name='conv5_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    # x = Convolution2D(512,3,3,activation='relu', border_mode='same', name='conv5_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    # x = Convolution2D(512,3,3,activation='linear', border_mode='same', name='conv5_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    # out5 = Convolution2D(1,1,1,activation='linear', border_mode='same', name='score-dsn5', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    # out5 = UpSampling2D(size=(16,16), name='upsample5')(out5)
    # out5 = Activation('sigmoid', name='sigmoid-dsn5')(out5)

    #Merge all outputs
    out = merge([out1,out2,out3], mode='concat', concat_axis=3)
    out = BatchNormalization()(out)
    out = Convolution2D(3,1,1,activation='relu', border_mode='same', name='new-score-weighting_pre', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(out)
    out = Convolution2D(1,1,1,activation='sigmoid', border_mode='same', name='new-score-weighting', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(out)

    #mask layer
    if mask:
    	m = Flatten()(inp)

    	for i in range(0,dense_layers):
    		m = Dense(dense_size, activation='relu',
            W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    	m = Dense(input_shape[0]*input_shape[1], activation='relu',
        W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    	m = Reshape(input_shape, name="mask")(m)

    	#merge
    	d = merge([out,m], mode='mul', name='merged')

        #finetune
        for i in range(0,num_conv):
            d = BatchNormalization()(d)
            d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu',
            border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)


        out = Convolution2D(1,Wfilter,Wfilter,activation='sigmoid',
        border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

    #weights to initialize final layer to be simple average
    arr = np.asarray([1.0/3,1.0/3,1.0/3])
    arr = arr.reshape((1,1,3,1))
    bias = np.asarray([0.0])

    model = Model(inp,[out,out1,out2,out3])
    #model.layers[-1].set_weights([arr,bias])
    #model.layers[-1].trainable = False
    return model

def hed_dense(input_shape=(64,64,1), Wfilter=3, Nfilters=32, dense_layers=1, dense_size=64,
num_conv=2, mask=True, l2_reg=0):
    inp = Input(shape=input_shape)

    #conv1
    x = Convolution2D(64,3,3,activation='relu', border_mode='same', name='conv1_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(inp)
    x = BatchNormalization()(x)
    x = Convolution2D(64,3,3,activation='linear', border_mode='same', name='conv1_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    out1 = Convolution2D(1,1,1,activation='sigmoid', border_mode='same', name='score-dsn1',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    #conv2
    x = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv2_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(128,3,3,activation='linear', border_mode='same', name='conv2_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    out2 = Convolution2D(1,1,1,activation='linear', border_mode='same', name='score-dsn2',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    out2 = UpSampling2D(size=(2,2), name='upsample2')(out2)
    out2 = Activation('sigmoid', name='sigmoid-dsn2')(out2)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    #conv3
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256,3,3,activation='linear', border_mode='same', name='conv3_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    out3 = Convolution2D(1,1,1,activation='linear', border_mode='same', name='score-dsn3',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    out3 = UpSampling2D(size=(4,4), name='upsample3')(out3)
    out3 = Activation('sigmoid', name='sigmoid-dsn3')(out3)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
    #
    # #conv4
    # x = Convolution2D(512,3,3,activation='relu', border_mode='same', name='conv4_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    # x = Convolution2D(512,3,3,activation='relu', border_mode='same', name='conv4_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    # x = Convolution2D(512,3,3,activation='linear', border_mode='same', name='conv4_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    # out4 = Convolution2D(1,1,1,activation='linear', border_mode='same', name='score-dsn4', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    # out4 = UpSampling2D(size=(8,8), name='upsample4')(out4)
    # out4 = Activation('sigmoid', name='sigmoid-dsn4')(out4)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)
    #
    # #conv5
    # x = Convolution2D(512,3,3,activation='relu', border_mode='same', name='conv5_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    # x = Convolution2D(512,3,3,activation='relu', border_mode='same', name='conv5_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    # x = Convolution2D(512,3,3,activation='linear', border_mode='same', name='conv5_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    # out5 = Convolution2D(1,1,1,activation='linear', border_mode='same', name='score-dsn5', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    # out5 = UpSampling2D(size=(16,16), name='upsample5')(out5)
    # out5 = Activation('sigmoid', name='sigmoid-dsn5')(out5)

    #Merge all outputs
    out = merge([out1,out2,out3], mode='concat', concat_axis=3)
    out = BatchNormalization()(out)
    out = Convolution2D(3,1,1,activation='relu', border_mode='same', name='new-score-weighting_pre', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(out)
    out = Convolution2D(1,1,1,activation='sigmoid', border_mode='same', name='new-score-weighting', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(out)


    m = Flatten()(out)

    for i in range(0,dense_layers):
    	m = Dense(dense_size, activation='relu',
        W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    m = Dense(input_shape[0]*input_shape[1], activation='relu',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    d = Reshape(input_shape, name="mask")(m)

    #merge
    #d = merge([out,m], mode='mul', name='merged')

    #finetune
    for i in range(0,num_conv):
        d = BatchNormalization()(d)
        d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu',
        border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)


    out = Convolution2D(1,Wfilter,Wfilter,activation='sigmoid',
    border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

    #weights to initialize final layer to be simple average
    arr = np.asarray([1.0/3,1.0/3,1.0/3])
    arr = arr.reshape((1,1,3,1))
    bias = np.asarray([0.0])

    model = Model(inp,[out,out1,out2,out3])
    #model.layers[-1].set_weights([arr,bias])
    #model.layers[-1].trainable = False
    return model

def I2INet(input_shape=(64,64,1), Wfilter=3, Nfilters=32, dense_layers=1, dense_size=64,
num_conv=2, mask=True, l2_reg=0, batchnorm=True):
    inp = Input(shape=input_shape)

    #conv1
    x = Convolution2D(64,3,3,activation='relu', border_mode='same', name='conv1_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(inp)
    if batchnorm: x = BatchNormalization()(x)
    out1 = Convolution2D(64,3,3,activation='relu', border_mode='same', name='conv1_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: out1 = BatchNormalization()(out1)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(out1)

    #conv2
    x = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv2_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: x = BatchNormalization()(x)
    out2 = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv2_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: out2 = BatchNormalization()(out2)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    #conv3
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: x = BatchNormalization()(x)
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: x = BatchNormalization()(x)
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)

    out3 = UpSampling2D(size=(2,2), name='upsample1')(x)

    #Second branch
    s = merge([out3,out2], mode='concat', concat_axis=3)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(256,1,1,activation='relu', border_mode='same', name='conv4_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv4_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv4_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    s_out_1 = Convolution2D(1,1,1,activation='sigmoid', border_mode='same', name='conv4_4', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)

    s = UpSampling2D(size=(2,2), name='upsample2')(s)
    s = merge([s,out1], mode='concat', concat_axis=3)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(128,1,1,activation='relu', border_mode='same', name='conv5_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(32,3,3,activation='relu', border_mode='same', name='conv5_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(32,3,3,activation='relu', border_mode='same', name='conv5_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    out = Convolution2D(1,1,1,activation='sigmoid', border_mode='same', name='conv5_4', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    #mask layer
    if mask:
    	m = Flatten()(inp)

    	for i in range(0,dense_layers):
    		m = Dense(dense_size, activation='relu',
            W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    	m = Dense(input_shape[0]*input_shape[1], activation='relu',
        W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    	m = Reshape(input_shape, name="mask")(m)

    	#merge
    	d = merge([out,m], mode='mul', name='merged')

        #finetune
        for i in range(0,num_conv):
            d = BatchNormalization()(d)
            d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu',
            border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

        out = Convolution2D(1,Wfilter,Wfilter,activation='sigmoid',
        border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

    #weights to initialize final layer to be simple average
    arr = np.asarray([1.0/3,1.0/3,1.0/3])
    arr = arr.reshape((1,1,3,1))
    bias = np.asarray([0.0])

    model = Model(inp,[out,s_out_1])
    #model.layers[-1].set_weights([arr,bias])
    #model.layers[-1].trainable = False
    return model

def I2INet_dense(input_shape=(64,64,1), Wfilter=3, Nfilters=32, dense_layers=1, dense_size=64,
num_conv=2, mask=True, l2_reg=0, batchnorm=True):
    inp = Input(shape=input_shape)

    #conv1
    x = Convolution2D(64,3,3,activation='relu', border_mode='same', name='conv1_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(inp)
    if batchnorm: x = BatchNormalization()(x)
    out1 = Convolution2D(64,3,3,activation='relu', border_mode='same', name='conv1_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: out1 = BatchNormalization()(out1)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(out1)

    #conv2
    x = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv2_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: x = BatchNormalization()(x)
    out2 = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv2_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: out2 = BatchNormalization()(out2)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    #conv3
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: x = BatchNormalization()(x)
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: x = BatchNormalization()(x)
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)

    out3 = UpSampling2D(size=(2,2), name='upsample1')(x)

    #Second branch
    s = merge([out3,out2], mode='concat', concat_axis=3)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(256,1,1,activation='relu', border_mode='same', name='conv4_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv4_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv4_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    s_out_1 = Convolution2D(1,1,1,activation='sigmoid', border_mode='same', name='conv4_4', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)

    s = UpSampling2D(size=(2,2), name='upsample2')(s)
    s = merge([s,out1], mode='concat', concat_axis=3)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(128,1,1,activation='relu', border_mode='same', name='conv5_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(32,3,3,activation='relu', border_mode='same', name='conv5_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(32,3,3,activation='relu', border_mode='same', name='conv5_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    i2i_out = Convolution2D(1,1,1,activation='sigmoid', border_mode='same', name='conv5_4', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    #mask layer

    m = Flatten()(i2i_out)

    for i in range(0,dense_layers):
    	m = Dense(dense_size, activation='relu',
        W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    m = Dense(input_shape[0]*input_shape[1], activation='relu',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    d = Reshape(input_shape, name="mask")(m)

    #merge
    #d = merge([out,m], mode='mul', name='merged')

    #finetune
    for i in range(0,num_conv):
        d = BatchNormalization()(d)
        d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu',
        border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

    out = Convolution2D(1,Wfilter,Wfilter,activation='sigmoid',
    border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

    #weights to initialize final layer to be simple average
    arr = np.asarray([1.0/3,1.0/3,1.0/3])
    arr = arr.reshape((1,1,3,1))
    bias = np.asarray([0.0])

    model = Model(inp,[out,s_out_1])
    #model.layers[-1].set_weights([arr,bias])
    #model.layers[-1].trainable = False
    return model

def I2INet_dense_mask(input_shape=(64,64,1), Wfilter=3, Nfilters=32, dense_layers=1, dense_size=64,
num_conv=2, mask=True, l2_reg=0, batchnorm=True):
    inp = Input(shape=input_shape)

    #conv1
    x = Convolution2D(64,3,3,activation='relu', border_mode='same', name='conv1_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(inp)
    if batchnorm: x = BatchNormalization()(x)
    out1 = Convolution2D(64,3,3,activation='relu', border_mode='same', name='conv1_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: out1 = BatchNormalization()(out1)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(out1)

    #conv2
    x = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv2_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: x = BatchNormalization()(x)
    out2 = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv2_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: out2 = BatchNormalization()(out2)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    #conv3
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: x = BatchNormalization()(x)
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: x = BatchNormalization()(x)
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)

    out3 = UpSampling2D(size=(2,2), name='upsample1')(x)

    #Second branch
    s = merge([out3,out2], mode='concat', concat_axis=3)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(256,1,1,activation='relu', border_mode='same', name='conv4_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv4_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv4_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    s_out_1 = Convolution2D(1,1,1,activation='sigmoid', border_mode='same', name='conv4_4', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)

    s = UpSampling2D(size=(2,2), name='upsample2')(s)
    s = merge([s,out1], mode='concat', concat_axis=3)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(128,1,1,activation='relu', border_mode='same', name='conv5_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(32,3,3,activation='relu', border_mode='same', name='conv5_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(32,3,3,activation='relu', border_mode='same', name='conv5_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    out = Convolution2D(1,1,1,activation='sigmoid', border_mode='same', name='conv5_4', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)

    #mask layer
    m = Flatten()(inp)

    for i in range(0,dense_layers):
    	m = Dense(dense_size, activation='relu',
        W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    m = Dense(input_shape[0]*input_shape[1], activation='relu',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    m = Reshape(input_shape, name="mask")(m)

    #merge
    d = merge([out,m], mode='mul', name='merged')

    #finetune
    for i in range(0,num_conv):
        d = BatchNormalization()(d)
        d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu',
        border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

    out = Convolution2D(1,Wfilter,Wfilter,activation='sigmoid',
    border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

    #FC out layer
    m = Flatten()(out)

    for i in range(0,dense_layers):
    	m = Dense(dense_size, activation='relu',
        W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    m = Dense(input_shape[0]*input_shape[1], activation='relu',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    d = Reshape(input_shape, name="mask_2")(m)

    #merge
    #d = merge([out,m], mode='mul', name='merged')

    #finetune
    for i in range(0,num_conv):
        d = BatchNormalization()(d)
        d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu',
        border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

    out = Convolution2D(1,Wfilter,Wfilter,activation='sigmoid',
    border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

    #weights to initialize final layer to be simple average
    arr = np.asarray([1.0/3,1.0/3,1.0/3])
    arr = arr.reshape((1,1,3,1))
    bias = np.asarray([0.0])

    model = Model(inp,[out,s_out_1])
    #model.layers[-1].set_weights([arr,bias])
    #model.layers[-1].trainable = False
    return model

def FCI2INet(input_shape=(64,64,1), Wfilter=3, Nfilters=32, dense_layers=1, dense_size=64,
num_conv=2, mask=True, l2_reg=0, batchnorm=True):
    inp = Input(shape=input_shape)

    #mask layer
    m = Flatten()(inp)

    for i in range(0,dense_layers):
    	m = Dense(dense_size, activation='relu',
        W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    m = Dense(input_shape[0]*input_shape[1], activation='relu',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    m = Reshape(input_shape, name="mask")(m)

    #conv1
    x = Convolution2D(64,3,3,activation='relu', border_mode='same', name='conv1_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)
    if batchnorm: x = BatchNormalization()(x)
    out1 = Convolution2D(64,3,3,activation='relu', border_mode='same', name='conv1_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: out1 = BatchNormalization()(out1)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(out1)

    #conv2
    x = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv2_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: x = BatchNormalization()(x)
    out2 = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv2_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: out2 = BatchNormalization()(out2)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    #conv3
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: x = BatchNormalization()(x)
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    if batchnorm: x = BatchNormalization()(x)
    x = Convolution2D(256,3,3,activation='relu', border_mode='same', name='conv3_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)

    out3 = UpSampling2D(size=(2,2), name='upsample1')(x)

    #Second branch
    s = merge([out3,out2], mode='concat', concat_axis=3)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(256,1,1,activation='relu', border_mode='same', name='conv4_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv4_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(128,3,3,activation='relu', border_mode='same', name='conv4_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    s_out_1 = Convolution2D(1,1,1,activation='sigmoid', border_mode='same', name='conv4_4', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)

    s = UpSampling2D(size=(2,2), name='upsample2')(s)
    s = merge([s,out1], mode='concat', concat_axis=3)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(128,1,1,activation='relu', border_mode='same', name='conv5_1', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(32,3,3,activation='relu', border_mode='same', name='conv5_2', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    if batchnorm: s = BatchNormalization()(s)
    s = Convolution2D(32,3,3,activation='relu', border_mode='same', name='conv5_3', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)
    out = Convolution2D(1,1,1,activation='sigmoid', border_mode='same', name='conv5_4', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(s)


    model = Model(inp,[out,s_out_1])
    #model.layers[-1].set_weights([arr,bias])
    #model.layers[-1].trainable = False
    return model

def FC_branch(input_shape=(64,64,1), Nfilters=32, Wfilter=3,num_conv_1=3, num_conv_2=3,
output_channels=1, mask=True, dense_layers=1,dense_size=64, obg=False, l2_reg=0.0):
    '''
    Makes an FCN neural network
    '''
    x = Input(shape=input_shape)

    #mask layer
    m = Flatten()(x)

    for i in range(0,dense_layers):
    	m = Dense(dense_size, activation='relu',
        W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    m = Dense(input_shape[0]*input_shape[1], activation='relu',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    d = Reshape(input_shape, name="mask")(m)

    #finetune
    for i in range(0,num_conv_2):
        d = BatchNormalization()(d)
        d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu',
        border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

    d = BatchNormalization()(d)
    d = Convolution2D(output_channels,
    Wfilter,Wfilter,activation='linear', border_mode='same',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)

    d = Activation('sigmoid', name='sigmoid')(d)
    FCN = Model(x,d)
    return FCN

def ConvFC(input_shape=(64,64,1), Nfilters=32, Wfilter=3,num_conv_1=3,
output_channels=1, dense_layers=1,dense_size=64, l2_reg=0.0):
    '''
    Makes an convolutional network folloed by FC layers
    '''
    x = Input(shape=input_shape)

    #main branch
    d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu',
    border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(x)
    d = BatchNormalization()(d)

    for i in range(0,num_conv_1):
        d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu',
        border_mode='same', W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(d)
        d = BatchNormalization()(d)

    dout = Convolution2D(1,Wfilter,Wfilter,activation='sigmoid', border_mode='same',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg), name='deepsupervision')(d)


    m = Flatten()(d)

    for i in range(0,dense_layers):
    	m = Dense(dense_size, activation='relu',
        W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)
        #m = BatchNormalization()(m)

    #m = Dense(input_shape[0]*input_shape[1], activation='linear',
    #W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    m = Dense(input_shape[0]*input_shape[1], activation='sigmoid',
    W_regularizer=l2(l2_reg), b_regularizer=l2(l2_reg))(m)

    m = Reshape(input_shape, name="out")(m)

    FCN = Model(x,[m,dout])
    return FCN
