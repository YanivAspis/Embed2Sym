import tensorflow as tf

def LeNet(output_size):
    return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=6, kernel_size=5),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=16, kernel_size=5),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=120, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(units=84, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(units=output_size, activation=tf.keras.activations.relu)
        ])



def ResNet56Cifar10(output_size=10, final_activation="softmax"):
    inputs = tf.keras.Input(shape=(32, 32, 3))

    conv_0_out = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same", use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(0.0001))(inputs)
    bn_0_out = tf.keras.layers.BatchNormalization()(conv_0_out)
    relu_0_out = tf.keras.layers.ReLU()(bn_0_out)

    block_input = relu_0_out
    for _ in range(9):
        block_conv_out_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same", use_bias=False,
                                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(block_input)
        block_bn_out_1 = tf.keras.layers.BatchNormalization()(block_conv_out_1)
        block_relu_out_1 = tf.keras.layers.ReLU()(block_bn_out_1)
        block_conv_out_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same", use_bias=False,
                                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
            block_relu_out_1)
        block_bn_out_2 = tf.keras.layers.BatchNormalization()(block_conv_out_2)
        block_skipped = block_input + block_bn_out_2
        block_relu_out_2 = tf.keras.layers.ReLU()(block_skipped)
        block_input = block_relu_out_2

    block_padded_out_1 = tf.keras.layers.ZeroPadding2D(padding=1)(block_input)
    block_conv_out_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="valid", use_bias=False,
                                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        block_padded_out_1)
    block_bn_out_1 = tf.keras.layers.BatchNormalization()(block_conv_out_1)
    block_relu_out_1 = tf.keras.layers.ReLU()(block_bn_out_1)
    block_conv_out_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", use_bias=False,
                                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        block_relu_out_1)
    block_bn_out_2 = tf.keras.layers.BatchNormalization()(block_conv_out_2)
    sliced_block_input = tf.keras.layers.Lambda(lambda x: x[:, ::2, ::2, :])(block_input)
    padded_block_input = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (8, 8)), data_format="channels_first")(
        sliced_block_input)
    block_skipped = padded_block_input + block_bn_out_2
    block_relu_out_2 = tf.keras.layers.ReLU()(block_skipped)
    block_input = block_relu_out_2

    for _ in range(8):
        block_conv_out_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", use_bias=False,
                                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(block_input)
        block_bn_out_1 = tf.keras.layers.BatchNormalization()(block_conv_out_1)
        block_relu_out_1 = tf.keras.layers.ReLU()(block_bn_out_1)
        block_conv_out_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", use_bias=False,
                                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
            block_relu_out_1)
        block_bn_out_2 = tf.keras.layers.BatchNormalization()(block_conv_out_2)
        block_skipped = block_input + block_bn_out_2
        block_relu_out_2 = tf.keras.layers.ReLU()(block_skipped)
        block_input = block_relu_out_2

    block_padded_out_1 = tf.keras.layers.ZeroPadding2D(padding=1)(block_input)
    block_conv_out_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="valid", use_bias=False,
                                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        block_padded_out_1)
    block_bn_out_1 = tf.keras.layers.BatchNormalization()(block_conv_out_1)
    block_relu_out_1 = tf.keras.layers.ReLU()(block_bn_out_1)
    block_conv_out_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", use_bias=False,
                                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        block_relu_out_1)
    block_bn_out_2 = tf.keras.layers.BatchNormalization()(block_conv_out_2)
    sliced_block_input = tf.keras.layers.Lambda(lambda x: x[:, ::2, ::2, :])(block_input)
    padded_block_input = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (16, 16)), data_format="channels_first")(
        sliced_block_input)
    block_skipped = padded_block_input + block_bn_out_2
    block_relu_out_2 = tf.keras.layers.ReLU()(block_skipped)
    block_input = block_relu_out_2

    for _ in range(8):
        block_conv_out_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", use_bias=False,
                                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(block_input)
        block_bn_out_1 = tf.keras.layers.BatchNormalization()(block_conv_out_1)
        block_relu_out_1 = tf.keras.layers.ReLU()(block_bn_out_1)
        block_conv_out_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", use_bias=False,
                                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
            block_relu_out_1)
        block_bn_out_2 = tf.keras.layers.BatchNormalization()(block_conv_out_2)
        block_skipped = block_input + block_bn_out_2
        block_relu_out_2 = tf.keras.layers.ReLU()(block_skipped)
        block_input = block_relu_out_2

    avg_pool_out = tf.keras.layers.GlobalAvgPool2D()(block_input)
    flattened = tf.keras.layers.Flatten()(avg_pool_out)
    dense_out = tf.keras.layers.Dense(units=output_size, kernel_regularizer=tf.keras.regularizers.l2(0.0001), bias_regularizer=tf.keras.regularizers.l2(0.0001))(flattened)
    outputs = {
        "softmax": tf.keras.layers.Softmax(),
        "relu": tf.keras.layers.ReLU()
    }[final_activation](dense_out)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def ComparisonNet(output_size):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units=20, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(units=16, activation=tf.keras.activations.relu)
    ])