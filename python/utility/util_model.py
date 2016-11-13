from keras.models import Model
from keras.layers import Input, Convolution2D, BatchNormalization, Dense, merge
from keras.layers import Reshape, Flatten, UpSampling2D
from keras.layers import MaxPooling2D, Activation
from keras.optimizers import Adam

def FCN(input_shape=(64,64,1), Nfilters=32, Wfilter=3,num_conv_1=3, num_conv_2=3,output_channels=1, mask=True, dense_layers=1,dense_size=64, obg=False):
    '''
    Makes an FCN neural network
    '''
    x = Input(shape=input_shape)

    #main branch
    d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(x)
    d = BatchNormalization(mode=2)(d)

    for i in range(0,num_conv_1):
        d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(d)
        d = BatchNormalization(mode=2)(d)

    d = Convolution2D(1,Wfilter,Wfilter,activation='relu', border_mode='same')(d)

    #mask layer
    if mask:
    	m = Flatten()(x)

    	for i in range(0,dense_layers):
    		m = Dense(dense_size, activation='relu')(m)

    	m = Dense(input_shape[0]*input_shape[1], activation='relu')(m)

    	m = Reshape(input_shape)(m)

    	#merge
    	d = merge([d,m], mode='mul')

    #finetune
    for i in range(0,num_conv_2):
        d = BatchNormalization(mode=2)(d)
        d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(d)

    d = BatchNormalization(mode=2)(d)
    d = Convolution2D(output_channels,
    Wfilter,Wfilter,activation='sigmoid', border_mode='same')(d)

    FCN = Model(x,d)

    if obg:
        d_out = Reshape(input_shape[0]*input_shape[1],output_channels)
        FCN_categorical = Model(x,d_out)
        return FCN,FCN_categorical
    else:
        return FCN

def OBG_FCN(FCN,OBP_FCN,input_shape=(64,64,1), Nfilters=32, Wfilter=3, output_channels=1):
    x = Input(shape=input_shape)

    fcn_out = FCN(x)

    obp_out = OBP_FCN(x)

    d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(obp_out)
    d = BatchNormalization(mode=2)(d)
    d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(d)
    d = BatchNormalization(mode=2)(d)
    d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(d)
    d = BatchNormalization(mode=2)(d)
    d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(d)
    d = BatchNormalization(mode=2)(d)
    d = Convolution2D(output_channels,Wfilter,Wfilter,activation='relu', border_mode='same')(d)

    #merge
    d = merge([d,fcn_out], mode='mul')
    OBG_FCN = model(x,d)
    return OBG_FCN

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
