import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Lambda, Conv2D, Conv2DTranspose, Activation, Reshape, concatenate, Concatenate, BatchNormalization, ZeroPadding2D, Input, Add, ReLU, GlobalAveragePooling2D, Multiply, Softmax
from keras.layers import UpSampling2D
from keras.regularizers import l2
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, Activation, Lambda, concatenate, AveragePooling2D, MaxPooling2D
from tensorflow.keras import backend as K

from resnet50 import ResNet50

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, DepthwiseConv2D, Lambda, AveragePooling2D, concatenate, LayerNormalization, MultiHeadAttention, Dropout, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def Upsample(tensor, size):
    '''bilinear upsampling'''
    name = tensor.name.split('/')[0] + '_upsample'

    def bilinear_upsample(x, size):
        resized = tf.image.resize(x, size)
        return resized
    y = Lambda(lambda x: bilinear_upsample(x, size), output_shape=size, name=name)(tensor)
    return y


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout_rate):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout_rate)(x, x)
    x = Dropout(dropout_rate)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(inputs.shape[-1])(x)
    x = Dropout(dropout_rate)(x)
    return x + res

def ASPP(tensor, transformer_layers=2, head_size=128, num_heads=4, ff_dim=512, dropout_rate=0.1):
    dims = K.int_shape(tensor)

    y_pool = AveragePooling2D(pool_size=(dims[1], dims[2]), name='average_pooling')(tensor)
    y_pool = Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal', name='pool_1x1conv2d', use_bias=False)(y_pool)
    y_pool = BatchNormalization(name='bn_pool')(y_pool)
    y_pool = Activation('relu', name='relu_pool')(y_pool)
    y_pool = Upsample(y_pool, size=[dims[1], dims[2]])

    y_1 = Conv2D(filters=128, kernel_size=1, dilation_rate=1, padding='same', kernel_initializer='he_normal', name='ASPP_conv2d_d1', use_bias=False)(tensor)
    y_1 = BatchNormalization(name='bn_1')(y_1)
    y_1 = Activation('relu', name='relu_1')(y_1)

    y_6 = depthwise_separable_conv2d(tensor, dilation_rate=6)
    y_12 = depthwise_separable_conv2d(tensor, dilation_rate=12)
    y_18 = depthwise_separable_conv2d(tensor, dilation_rate=18)

    y = concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat')

    batch_size, height, width, channels = y.shape
    y = Reshape((height * width, channels))(y) 
    
    for _ in range(transformer_layers):
        y = transformer_encoder(y, head_size, num_heads, ff_dim, dropout_rate)

    y = Reshape((height, width, channels))(y)
    
    y = Conv2D(filters=128, kernel_size=1, dilation_rate=1, padding='same', kernel_initializer='he_normal', name='ASPP_conv2d_final_transformer', use_bias=False)(y)
    y = BatchNormalization(name='bn_final_transformer')(y)
    y = Activation('relu', name='relu_final_transformer')(y)
    
    return y


def depthwise_separable_conv2d(x, dilation_rate):
    y = DepthwiseConv2D(kernel_size=3, dilation_rate=dilation_rate, padding='same', depthwise_initializer='he_normal', use_bias=False)(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal', use_bias=False)(y)
    y = BatchNormalization()(y)
    return Activation('relu')(y)

def multi_scale_pooling(x, scales=[1, 2, 3], img_height=None, img_width=None):
    pooled_features = [x]
    for scale in scales:
        pool_size = (scale, scale)
        pooled = AveragePooling2D(pool_size=pool_size, strides=pool_size, padding='same')(x)
        upsampled = UpSampling2D(size=pool_size)(pooled)
        pooled_features.append(upsampled)
    return concatenate(pooled_features)


def upsample_and_add(x1, x2):
    x1 = UpSampling2D(size=(x2.shape[1] // x1.shape[1], x2.shape[2] // x1.shape[2]))(x1)
    if x1.shape[-1] != x2.shape[-1]:
        x1 = Conv2D(filters=x2.shape[-1], kernel_size=1, padding='same')(x1)
    return Add()([x1, x2])


def ResidualDepthwiseConv2D(inputs, filters, kernel_size=3, strides=1, dilation_rate=1, pool_sizes=[1, 3, 5]):
    depthwise = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate, padding='same')(inputs)
    depthwise = BatchNormalization()(depthwise)
    depthwise = ReLU()(depthwise)

    pointwise = Conv2D(filters, kernel_size=1, strides=1, padding='same')(depthwise)
    pointwise = BatchNormalization()(pointwise)
    pointwise = ReLU()(pointwise)
    
    
    if strides != 1 or K.int_shape(inputs)[-1] != filters:
        inputs = Conv2D(filters, kernel_size=1, strides=strides, padding='same')(inputs)
        inputs = BatchNormalization()(inputs)

    out = Add()([inputs, pointwise])
    return ReLU()(out)

def edge_detection_module(input_tensor):
    x = Conv2D(64, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    return x
# def edge_detection_module(input_tensor):
#     x = DepthwiseConv2D((3, 3), padding='same', use_bias=False)(input_tensor)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
    
#     se = GlobalAveragePooling2D()(x)
#     se = Reshape((1, 1, -1))(se)
#     se = Dense(x.shape[-1] // 16, activation='relu', use_bias=False)(se)
#     se = Dense(x.shape[-1], activation='sigmoid', use_bias=False)(se)
    
#     x = multiply([x, se])
#     x = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
#     return x

def context_enhancement_module(input_feature, out_channels, dilation_rates=[1, 6, 12]):
    dilated_features = [Conv2D(out_channels, 3, padding='same', dilation_rate=rate, activation='relu')(input_feature) for rate in dilation_rates]

    concatenated_features = Concatenate(axis=-1)(dilated_features)
    fusion_weights = Conv2D(out_channels, 1, activation='sigmoid')(concatenated_features)
    fused_feature = Add()([Multiply()([fusion_weights, feature]) for feature in dilated_features])

    global_context = GlobalAveragePooling2D()(input_feature)
    global_context = Dense(out_channels, activation='relu')(global_context)
    global_context = Dense(out_channels, activation='sigmoid')(global_context)
    global_context = Reshape((1, 1, out_channels))(global_context)

    attention_weighted_feature = Multiply()([fused_feature, global_context])
    return attention_weighted_feature


def MY_MODEL(img_height, img_width, nclasses=19):
    print('*** Building MY_MODEL Network ***')
    shared_input = Input(shape=(img_height, img_width, 3))

    base_model = ResNet50(input_tensor=shared_input, weights='imagenet', include_top=False)
    
    image_features = base_model.get_layer('activation_39').output
    
    x_a = ASPP(image_features)
    x_a = Upsample(tensor=x_a, size=[img_height // 4, img_width // 4])

    x_b = base_model.get_layer('activation_9').output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same', kernel_initializer='he_normal', name='low_level_projection', use_bias=False)(x_b)
    x_b = BatchNormalization(name=f'bn_low_level_projection')(x_b)
    x_b = Activation('relu', name='low_level_activation')(x_b)
    
    feature_layer = base_model.get_layer('activation_23').output
    cem_feature = context_enhancement_module(feature_layer, out_channels=256)
    x_c = Upsample(cem_feature, size=[img_height // 4, img_width // 4])
    
    edge_feature = edge_detection_module(shared_input)
    edge_feature_upsampled = Upsample(edge_feature, size=[img_height // 4, img_width // 4])
    
    x_b_multi_scale = multi_scale_pooling(x_b, scales=[1, 2, 4], img_height=img_height//4, img_width=img_width//4)

    
    x_a_combined = concatenate([x_a, edge_feature_upsampled], axis=-1)
    x_b_combined = concatenate([x_b, x_b_multi_scale], axis=-1)

    x = concatenate([x_a_combined, x_b_combined, x_c], name='decoder_concat')
    

    x = ResidualDepthwiseConv2D(x, filters=128, kernel_size=3, strides=1, dilation_rate=1)
    x = ResidualDepthwiseConv2D(x, filters=128, kernel_size=3, strides=1, dilation_rate=2) 
    
    
    x = Upsample(x, [img_height, img_width])

    x = Conv2D(nclasses, (1, 1), name='output_layer')(x)

    model = Model(inputs=base_model.input, outputs=x, name='MY_MODEL')
    print(f'*** Output_Shape => {model.output_shape} ***')
    return model
