import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply, Add, Concatenate, Activation, BatchNormalization, MaxPooling2D, Input, DepthwiseConv2D, Lambda, Reshape, UpSampling2D, Dropout, Flatten
from keras.regularizers import l2
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall


def channel_attention_layer(inputs, channels, reduction=8):
    avg_pool = GlobalAveragePooling2D()(inputs)
    max_pool = GlobalMaxPooling2D()(inputs)
    fc1 = Dense(channels // reduction, activation='relu')(avg_pool)
    fc2 = Dense(channels)(fc1)
    fc3 = Dense(channels // reduction, activation='relu')(max_pool)
    fc4 = Dense(channels)(fc3)
    add = Add()([fc2, fc4])
    scale = Activation('sigmoid')(add)
    output = Multiply()([inputs, scale])
    return output


def spatial_attention_layer(inputs):
    avg_out = tf.reduce_mean(inputs, axis=3, keepdims=True)
    max_out = tf.reduce_max(inputs, axis=3, keepdims=True)
    x = Concatenate(axis=3)([avg_out, max_out])
    scale = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(x)
    output = Multiply()([inputs, scale])
    return output


def cbam_module(inputs, channels, reduction=8):
    x = channel_attention_layer(inputs, channels, reduction)
    x = spatial_attention_layer(x)
    return x


# def noah_CNN(input_tensor):
#     # mobilenet_base = MobileNet(weights='imagenet', include_top=False, input_tensor=input_tensor)
#     # for layer in mobilenet_base.layers:
#     #     layer.trainable = False

#     # mobilenet_output = mobilenet_base.output

#     x = Conv2D(32, (1, 1))(input_tensor)
#     x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
#     x = MaxPooling2D((2, 2))(x)

#     x = Conv2D(64, (3, 3))(x)
#     x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
#     x = MaxPooling2D((2, 2))(x)

#     x = Conv2D(64, (3, 3))(x)
#     x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
#     x = MaxPooling2D((2, 2))(x)

#     dilation = Conv2D(32, (1, 1), dilation_rate=2)(x)
#     dilation = tf.keras.layers.LeakyReLU(alpha=0.01)(dilation)
#     dilation = MaxPooling2D((2, 2))(dilation)

#     x = Conv2D(32, (1, 1))(x)
#     x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
#     x = MaxPooling2D((2, 2))(x)

#     pool1 = GlobalAveragePooling2D()(x)
#     pool1 = Reshape((1, 1, 32))(pool1)
#     pool1 = UpSampling2D(size=(x.shape[1] // pool1.shape[1], x.shape[2] // pool1.shape[2]))(pool1)

#     x = Concatenate(axis=-1)([dilation, pool1])

#     x = cbam_module(x, channels=64)

#     output = Flatten()(x)
#     return output


def residual_block_with_dwc(inputs, units, adjust_shortcut=False, pool_sizes=[1, 3, 5]):
    convModule = [Conv2D(units, (size, size), padding='same', activation='relu')(inputs) for size in pool_sizes]
    multiScaleFeatures = Concatenate()(convModule)

    poolModule = [MaxPooling2D((size, size), strides=(1, 1), padding='same')(multiScaleFeatures) for size in pool_sizes]
    poolFeatures = Concatenate()(poolModule)

    fusion = Dense(units, activation='relu')(poolFeatures)

    convDWC = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1)(poolFeatures)
    convPWC = Conv2D(units, kernel_size=(1, 1), padding='same')(convDWC)

    DWCWithFusion = Add()([convPWC, fusion])

    if adjust_shortcut:
        shortcut_conv = Conv2D(units, kernel_size=3, padding="same", kernel_regularizer=l2(0.01))(inputs)
        shortcut = shortcut_conv
    else:
        shortcut = inputs

    out = Add()([DWCWithFusion, shortcut])
    out = tf.nn.leaky_relu(out)
    return out


def model_with_residual(inputs):
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D()(x)

    for i, filters in enumerate([64, 128, 256, 512]):
        adjust_shortcut = (i != 0)
        x = residual_block_with_dwc(x, filters, adjust_shortcut=adjust_shortcut)
        x = MaxPooling2D()(x)
        # x = cbam_module(x, channels=filters)

    final_output = Flatten()(x)
    return final_output


def create_combined_model(input_layer):
    # input_layer = Input(shape=(224, 224, 3))

    output = model_with_residual(input_layer)
    # output2 = noah_CNN(input_layer)

#     def apply_weight1(x):
#         return x * 0.5

#     def apply_weight2(x):
#         return x * 0.5

    # weighted_output1 = Lambda(apply_weight1)(output1)
    # weighted_output2 = Lambda(apply_weight2)(output2)

    # merged_output = Concatenate()([weighted_output1, weighted_output2])

    x = Dense(64)(output)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    final_output = Dense(9, activation='softmax')(x)

    combined_model = Model(inputs=input_layer, outputs=final_output)
    return combined_model

# def get_combined_model():
#     input_layer = Input(shape=(224, 224, 3))
#     # model1 = Model(inputs=input_layer, outputs=model_with_residual(input_layer))
#     # model2 = Model(inputs=input_layer, outputs=noah_CNN(input_layer))
#     combined_model = create_combined_model()
#     combined_model.summary()

