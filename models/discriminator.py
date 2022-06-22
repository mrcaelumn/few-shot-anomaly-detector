import tensorflow as tf

# create discriminator model
def build_discriminator(inputs, img_size=128):
    num_layers = 4
    if img_size > 128:
        num_layers = 5
    f = [2**i for i in range(num_layers)]
    x = inputs
    features = []
    for i in range(0, num_layers):
        if i == 0:
            x = tf.keras.layers.DepthwiseConv2D(kernel_size = (3, 3), strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.Conv2D(f[i] * img_size ,kernel_size = (1, 1),strides=(2,2), padding='same')(x)
            # x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
        
        else:
            x = tf.keras.layers.DepthwiseConv2D(kernel_size = (3, 3), strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.Conv2D(f[i] * img_size ,kernel_size = (1, 1),strides=(2,2), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
        # x = tf.keras.layers.Dropout(0.3)(x)
        
        features.append(x)
           
    x = tf.keras.layers.Flatten()(x)
    features.append(x)
    output = tf.keras.layers.Dense(1, activation="softmax")(x)

    model = tf.keras.models.Model(inputs, outputs = [features, output], name="discriminator")
    
    return model