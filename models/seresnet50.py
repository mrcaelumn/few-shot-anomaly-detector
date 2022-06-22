import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, MaxPool2D, \
GlobalAveragePooling2D, Conv2DTranspose, Concatenate, Input, Dense, Reshape, Multiply, add, Flatten, ZeroPadding2D
from tensorflow.keras.models import Model
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.utils.layer_utils import get_source_inputs
from keras import backend as K


def conv_block_2nd(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # x = Activation(tf.nn.leaky_relu)(x)
 
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # x = Activation(tf.nn.leaky_relu)(x)
    
    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block_2nd(x, num_filters)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    bn_eps = 0.0001
        
    block_name = str(stage) + "_" + str(block)
    conv_name_base = "conv" + block_name
    relu_name_base = "relu" + block_name


    x = Conv2D(filters1, (1, 1), use_bias=False, name=conv_name_base + '_x1')(input_tensor)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x1_bn')(x)
    # x = Activation('relu', name=relu_name_base + '_x1')(x)
    x = Activation(tf.nn.leaky_relu, name=relu_name_base + '_x1')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=False, name=conv_name_base + '_x2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x2_bn')(x)
    # x = Activation('relu', name=relu_name_base + '_x2')(x)
    x = Activation(tf.nn.leaky_relu, name=relu_name_base + '_x2')(x)

    x = Conv2D(filters3, (1, 1), use_bias=False, name=conv_name_base + '_x3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x3_bn')(x)

    se = GlobalAveragePooling2D(name='pool' + block_name + '_gap')(x)
    se = Dense(filters3 // 8, activation='relu', name = 'fc' + block_name + '_sqz')(se)
    se = Dense(filters3, activation='sigmoid', name = 'fc' + block_name + '_exc')(se)
    se = Reshape([1, 1, filters3])(se)
    x = Multiply(name='scale' + block_name)([x, se])

    x = add([x, input_tensor], name='block_' + block_name + '_x4')
    # x = Activation('relu', name='block_out_' + block_name + '_x4')(x)
    x = Activation(tf.nn.leaky_relu, name='block_out_' + block_name + '_x4')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    bn_eps = 0.0001
    
    block_name = str(stage) + "_" + str(block)
    conv_name_base = "conv" + block_name
    relu_name_base = "relu" + block_name

    x = Conv2D(filters1, (1, 1), use_bias=False, name=conv_name_base + '_x1')(input_tensor)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x1_bn')(x)
    # x = Activation('relu', name=relu_name_base + '_x1')(x)
    x = Activation(tf.nn.leaky_relu, name=relu_name_base + '_x1')(x)

    x = Conv2D(filters2, kernel_size, strides=strides, padding='same', use_bias=False, name=conv_name_base + '_x2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x2_bn')(x)
    # x = Activation('relu', name=relu_name_base + '_x2')(x)
    x = Activation(tf.nn.leaky_relu, name=relu_name_base + '_x2')(x)

    x = Conv2D(filters3, (1, 1), use_bias=False, name=conv_name_base + '_x3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x3_bn')(x)
    
    se = GlobalAveragePooling2D(name='pool' + block_name + '_gap')(x)
    se = Dense(filters3 // 16, activation='relu', name = 'fc' + block_name + '_sqz')(se)
    se = Dense(filters3, activation='sigmoid', name = 'fc' + block_name + '_exc')(se)
    se = Reshape([1, 1, filters3])(se)
    x = Multiply(name='scale' + block_name)([x, se])
    
    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=False, name=conv_name_base + '_prj')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_prj_bn')(shortcut)

    x = add([x, shortcut], name='block_' + block_name)
    # x = Activation('relu', name='block_out_' + block_name)(x)
    x = Activation(tf.nn.leaky_relu, name='block_out_' + block_name)(x)
    return x


def SEResNet50(include_top=True, weights='imagenet',
               input_tensor=None, input_shape=None,
               pooling=None,
               classes=1000):

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=225,
                                      min_size=160,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape, name="input_1")
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape, name="input_1")
        else:
            img_input = input_tensor
            
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    bn_eps = 0.0001
    
    # x = ZeroPadding2D(padding=(2, 2), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', use_bias=False, name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    # x = Activation(tf.nn.leaky_relu, name='conv1_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='conv1_pool')(x)
    # x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(x)
    
    x = conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

    # x = Flatten()(x)
    # x = Dense(classes, activation='softmax', name='fc6')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='se-resnet50')
    return model  

def build_seresnet50_unet(input_shape, img_size=128, img_channel=3):
    inputs = Input(input_shape, name="input_1")
    """ Pre-trained ResNet50 Model """
    seresnet50 = SEResNet50(weights=None, input_tensor=inputs)
    # seresnet50.summary()
    """ Encoder """
    s1 = seresnet50.get_layer("input_1").output           ## (512 x 512)
    s2 = seresnet50.get_layer("conv1_relu").output        ## (256 x 256)
    s3 = seresnet50.get_layer("relu3_1_x1").output  ## (128 x 128)
    s4 = seresnet50.get_layer("relu4_1_x1").output  ## (64 x 64)
    s5 = seresnet50.get_layer("relu5_1_x1").output  ## (32 x 32)

    """ Bridge """
    b1 = seresnet50.get_layer("block_out_5_3_x4").output  ## (16 x 16)

    """ Decoder """
    x = img_size
    d1 = decoder_block(b1, s5, x)                     ## (32 x 32)
    x = x/2
    d2 = decoder_block(d1, s4, x)                     ## (64 x 64)
    x = x/2
    d3 = decoder_block(d2, s3, x)                     ## (128 x 128)
    x = x/2
    d4 = decoder_block(d3, s2, x)                      ## (256 x 256)
    x = x/2
    d5 = decoder_block(d4, s1, x)                      ## (512 x 512)
    
    """ Output """
    outputs = tf.keras.layers.Conv2D(img_channel, 1, padding="same", activation="tanh")(d5)

    model = tf.keras.models.Model(inputs, outputs, name="SEResNet50_U-Net")

    return model