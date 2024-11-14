import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, \
    GlobalAveragePooling2D, Multiply, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation
import hfloss

# Squeeze-and-Excitation Block
def squeeze_and_excitation_block(input_tensor, reduction_ratio=16):
    channel = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(channel // reduction_ratio, activation='relu')(se)
    se = Dense(channel, activation='sigmoid')(se)
    se = tf.reshape(se, [-1, 1, 1, channel])
    se = Multiply()([input_tensor, se])
    return se

# SERB Block
def serb_block(input_tensor, filters):
    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_and_excitation_block(x)

    if input_tensor.shape[-1] != filters:
        input_tensor = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=l2(1e-4))(input_tensor)

    x = Add()([input_tensor, x])
    return x

# ASPP Block
def aspp_block(input_tensor, filters, rates=[1, 6, 12]):
    aspp_outputs = []
    for rate in rates:
        aspp = Conv2D(filters, (3, 3), dilation_rate=rate, padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
        aspp = BatchNormalization()(aspp)
        aspp = Activation('relu')(aspp)
        aspp_outputs.append(aspp)

    merged_aspp = Concatenate()(aspp_outputs)
    output = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=l2(1e-4))(merged_aspp)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    return output

def unwrapnet(input_shape=(256, 256, 1)):
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    merged_inputs = Concatenate()([input1, input2])

    c1 = serb_block(merged_inputs, 64)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = serb_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = serb_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = serb_block(p3, 512)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = serb_block(p4, 512)

    aspp = aspp_block(c5, 512)

    u6 = UpSampling2D((2, 2))(aspp)
    u6 = Concatenate()([u6, c4])
    c6 = serb_block(u6, 256)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c3])
    c7 = serb_block(u7, 128)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = Concatenate()([u8, c2])
    c8 = serb_block(u8, 64)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate()([u9, c1])
    c9 = serb_block(u9, 64)

    outputs = Conv2D(1, (1, 1), activation='linear')(c9)

    model = Model([input1, input2], outputs)
    model.compile(optimizer='adam', loss=hfloss.high_fidelity_loss, metrics=['mae'])

    return model