import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Lambda, Activation, MaxPooling2D, MaxPool2D, \
GlobalAveragePooling2D, Conv2DTranspose, Concatenate, \
Input, Dense, Reshape, Multiply, Add, Flatten, ZeroPadding2D
from tensorflow.keras.models import Model
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.utils.layer_utils import get_source_inputs
from keras import backend

import keras_applications as ka

import collections

IMG_SIZE = 512

ModelParams = collections.namedtuple(
    'ModelParams',
    ['model_name', 'repetitions', 'residual_block', 'groups',
     'reduction', 'init_filters', 'input_3x3', 'dropout']
)


def get_bn_params(**params):
    axis = 3 if backend.image_data_format() == 'channels_last' else 1
    default_bn_params = {
        'axis': axis,
        'epsilon': 9.999999747378752e-06,
    }
    default_bn_params.update(params)
    return default_bn_params


def get_num_channels(tensor):
    channels_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    return backend.int_shape(tensor)[channels_axis]

def expand_dims(x, channels_axis):
    if channels_axis == 3:
        return x[:, None, None, :]
    elif channels_axis == 1:
        return x[:, :, None, None]
    else:
        raise ValueError("Slice axis should be in (1, 3), got {}.".format(channels_axis))
        
def conv_block_2nd(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block_2nd(x, num_filters)
    return x

def slice_tensor(x, start, stop, axis):
    if axis == 3:
        return x[:, :, :, start:stop]
    elif axis == 1:
        return x[:, start:stop, :, :]
    else:
        raise ValueError("Slice axis should be in (1, 3), got {}.".format(axis))

def GroupConv2D(filters,
                kernel_size,
                strides=(1, 1),
                groups=32,
                kernel_initializer='he_uniform',
                use_bias=True,
                activation='linear',
                padding='valid',
                **kwargs):
    """
    Grouped Convolution Layer implemented as a Slice,
    Conv2D and Concatenate layers. Split filters to groups, apply Conv2D and concatenate back.
    Args:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer, specifying the stride
            length of the convolution.
        groups: Integer, number of groups to split input filters to.
        kernel_initializer: Regularizer function applied to the kernel weights matrix.
        use_bias: Boolean, whether the layer uses a bias vector.
        activation: Activation function to use (see activations).
            If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
        padding: one of "valid" or "same" (case-insensitive).
    Input shape:
        4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last".
    Output shape:
        4D tensor with shape: (batch, new_rows, new_cols, filters) if data_format is "channels_last".
        rows and cols values might have changed due to padding.
    """

    slice_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor):
        inp_ch = int(backend.int_shape(input_tensor)[-1] // groups)  # input grouped channels
        out_ch = int(filters // groups)  # output grouped channels

        blocks = []
        for c in range(groups):
            slice_arguments = {
                'start': c * inp_ch,
                'stop': (c + 1) * inp_ch,
                'axis': slice_axis,
            }
            x = Lambda(slice_tensor, arguments=slice_arguments)(input_tensor)
            x = Conv2D(out_ch,
                              kernel_size,
                              strides=strides,
                              kernel_initializer=kernel_initializer,
                              use_bias=use_bias,
                              activation=activation,
                              padding=padding)(x)
            blocks.append(x)

        x = Concatenate(axis=slice_axis)(blocks)
        return x

    return layer

def ResNeXt(
        model_params,
        include_top=True,
        input_tensor=None,
        input_shape=None,
        classes=1000,
        weights='imagenet',
        **kwargs):
    """Instantiates the ResNet, SEResNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    Args:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    Returns:
        A Keras model instance.
    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()

    # resnext bottom
    x = BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
    x = BatchNormalization(name='bn0', **bn_params)(x)
    # x = Activation('relu', name='relu0')(x)
    x = Activation(tf.nn.leaky_relu)(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)

    # resnext body
    init_filters = 128
    for stage, rep in enumerate(model_params.repetitions):
        for block in range(rep):

            filters = init_filters * (2 ** stage)

            # first block of first stage without strides because we have maxpooling before
            if stage == 0 and block == 0:
                x = conv_block(filters, stage, block, strides=(1, 1), **kwargs)(x)

            elif block == 0:
                x = conv_block(filters, stage, block, strides=(2, 2), **kwargs)(x)

            else:
                x = identity_block(filters, stage, block, **kwargs)(x)

    # resnext top
    if include_top:
        x = GlobalAveragePooling2D(name='pool1')(x)
        x = Dense(classes, name='fc1')(x)
        # x = Activation('softmax', name='softmax')(x)
        x = Activation('tanh', name='softmax')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model
    model = Model(inputs, x)

    if weights:
        if type(weights) == str and os.path.exists(weights):
            model.load_weights(weights)
        else:
            load_model_weights(model, model_params.model_name,
                               weights, classes, include_top, **kwargs)

    return model


def ChannelSE(reduction=16, **kwargs):
    """
    Squeeze and Excitation block, reimplementation inspired by
        https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py
    Args:
        reduction: channels squeeze factor
    """
    channels_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor):
        # get number of channels/filters
        channels = backend.int_shape(input_tensor)[channels_axis]

        x = input_tensor

        # squeeze and excitation block in PyTorch style with
        x = GlobalAveragePooling2D()(x)
        x = Lambda(expand_dims, arguments={'channels_axis': channels_axis})(x)
        x = Conv2D(channels // reduction, (1, 1), kernel_initializer='he_uniform')(x)
        # x = Activation('relu')(x)
        x = Activation(tf.nn.leaky_relu)(x)
        x = Conv2D(channels, (1, 1), kernel_initializer='he_uniform')(x)
        # x = Activation('sigmoid')(x)
        x = Activation('tanh')(x)

        # apply attention
        x = Multiply()([input_tensor, x])

        return x

    return layer

def SEResNeXtBottleneck(filters, reduction=16, strides=1, groups=32, base_width=4, **kwargs):
    bn_params = get_bn_params()

    def layer(input_tensor):
        x = input_tensor
        residual = input_tensor

        width = (filters // 4) * base_width * groups // 64

        # bottleneck
        x = Conv2D(width, (1, 1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = BatchNormalization(**bn_params)(x)
        # x = Activation('relu')(x)
        x = Activation(tf.nn.leaky_relu)(x)

        x = ZeroPadding2D(1)(x)
        x = GroupConv2D(width, (3, 3), strides=strides, groups=groups,
                        kernel_initializer='he_uniform', use_bias=False, **kwargs)(x)
        x = BatchNormalization(**bn_params)(x)
        # x = Activation('relu')(x)
        x = Activation(tf.nn.leaky_relu)(x)

        x = Conv2D(filters, (1, 1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = BatchNormalization(**bn_params)(x)

        #  if number of filters or spatial dimensions changed
        #  make same manipulations with residual connection
        x_channels = get_num_channels(x)
        r_channels = get_num_channels(residual)

        if strides != 1 or x_channels != r_channels:
            residual = Conv2D(x_channels, (1, 1), strides=strides,
                                     kernel_initializer='he_uniform', use_bias=False)(residual)
            residual = BatchNormalization(**bn_params)(residual)

        # apply attention module
        x = ChannelSE(reduction=reduction, **kwargs)(x)

        # add residual connection
        x = Add()([x, residual])

        # x = Activation('relu')(x)
        x = Activation(tf.nn.leaky_relu)(x)

        return x

    return layer

def SEResNext50(
        model_params,
        input_tensor=None,
        input_shape=None,
        include_top=True,
        classes=1000,
        weights='imagenet',
        **kwargs
):
    """Instantiates the ResNet, SEResNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    Args:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    Returns:
        A Keras model instance.
    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """


    residual_block = model_params.residual_block
    init_filters = model_params.init_filters
    bn_params = get_bn_params()

    # define input
    if input_tensor is None:
        input = Input(shape=input_shape, name='input')
    else:
        if not backend.is_keras_tensor(input_tensor):
            input = Input(tensor=input_tensor, shape=input_shape)
        else:
            input = input_tensor

    x = input

    if model_params.input_3x3:

        x = ZeroPadding2D(1)(x)
        x = Conv2D(init_filters, (3, 3), strides=2,
                          use_bias=False, kernel_initializer='he_uniform')(x)
        x = BatchNormalization(**bn_params)(x)
        # x = Activation('relu')(x)
        x = Activation(tf.nn.leaky_relu)(x)

        x = ZeroPadding2D(1)(x)
        x = Conv2D(init_filters, (3, 3), use_bias=False,
                          kernel_initializer='he_uniform')(x)
        x = BatchNormalization(**bn_params)(x)
        # x = Activation('relu')(x)
        x = Activation(tf.nn.leaky_relu)(x)

        x = ZeroPadding2D(1)(x)
        x = Conv2D(init_filters * 2, (3, 3), use_bias=False,
                          kernel_initializer='he_uniform')(x)
        x = BatchNormalization(**bn_params)(x)
        # x = Activation('relu')(x)
        x = Activation(tf.nn.leaky_relu)(x)

    else:
        x = ZeroPadding2D(3)(x)
        x = Conv2D(init_filters, (7, 7), strides=2, use_bias=False,
                          kernel_initializer='he_uniform')(x)
        x = BatchNormalization(**bn_params)(x)
        # x = Activation('relu')(x)
        x = Activation(tf.nn.leaky_relu)(x)

    x = ZeroPadding2D(1)(x)
    x = MaxPooling2D((3, 3), strides=2)(x)

    # body of resnet
    filters = model_params.init_filters * 2
    for i, stage in enumerate(model_params.repetitions):
        # increase number of filters with each stage
        filters *= 2

        for j in range(stage):
            # decrease spatial dimensions for each stage (except first, because we have maxpool before)
            if i == 0 and j == 0:
                x = residual_block(filters, reduction=model_params.reduction,
                                   strides=1, groups=model_params.groups, is_first=True, **kwargs)(x)

            elif i != 0 and j == 0:
                x = residual_block(filters, reduction=model_params.reduction,
                                   strides=2, groups=model_params.groups, **kwargs)(x)
            else:
                x = residual_block(filters, reduction=model_params.reduction,
                                   strides=1, groups=model_params.groups, **kwargs)(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        if model_params.dropout is not None:
            x = Dropout(model_params.dropout)(x)
        x = Dense(classes)(x)
        # x = Activation('softmax', name='output')(x)
        x = Activation('tanh', name='output')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = input

    model = Model(inputs, x, name="SEResNext50")

    if weights:
        if type(weights) == str and os.path.exists(weights):
            model.load_weights(weights)
        else:
            load_model_weights(model, model_params.model_name,
                               weights, classes, include_top, **kwargs)

    return model


def build_seresnext50_unet(input_shape, img_size=128, img_channel=3):
    inputs = Input(input_shape, name="input_1")
    MODEL_PARS = ModelParams(
        'seresnext50', repetitions=(3, 4, 6, 3), residual_block=SEResNeXtBottleneck,
        groups=32, reduction=16, init_filters=64, input_3x3=False, dropout=None,
    )
    seresnext50 = SEResNext50(MODEL_PARS, weights=None, input_tensor=inputs)
    
    # seresnext50.summary()
    # for idx, layer in enumerate(seresnext50.layers):
    #     print(idx, layer.name, layer.output.type_spec.shape)
        
    """ Encoder """
    s1 = seresnext50.get_layer(index=0).output           ## (512 x 512)
    s2 = seresnext50.get_layer(index=4).output        ## (256 x 256)
    s3 = seresnext50.get_layer(index=257).output  ## (128 x 128)
    s4 = seresnext50.get_layer(index=587).output  ## (64 x 64)
    s5 = seresnext50.get_layer(index=1081).output  ## (32 x 32)

    """ Bridge """
    b1 = seresnext50.get_layer(index=1326).output  ## (16 x 16)

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
    outputs = Conv2D(img_channel, 1, padding="same", activation="tanh")(d5)

    model = Model(inputs, outputs, name="SEResNext50_U-Net")
    
    return model