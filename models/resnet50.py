import tensorflow as tf


def conv_block(input, num_filters, bn=True):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding="same")(input)
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x

def decoder_block(input, skip_features, num_filters, bn=True):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters, bn)
    # x = conv_block(x, num_filters, bn)
    return x

# create generator model based on resnet50 and unet network
def build_generator_resnet50_unet(input_shape, img_size=128, img_channel=3):
    # print(inputs)
    inputs = tf.keras.layers.Input(input_shape, name="input_1")
    # print("pretained start")
    """ Pre-trained ResNet50 Model """
    resnet50 = tf.keras.applications.ResNet50(include_top=True, weights="imagenet", input_tensor=inputs)
    
    """ Encoder """
    s1 = resnet50.get_layer("input_1").output           ## (512 x 512)
    s2 = resnet50.get_layer("conv1_relu").output        ## (256 x 256)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)
    s5 = resnet50.get_layer("conv4_block6_out").output  ## (32 x 32)

    """ Bridge """
    b1 = resnet50.get_layer("conv5_block3_out").output  ## (16 x 16)

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

    model = tf.keras.models.Model(inputs, outputs, name="ResNet50_U-Net")

    return model