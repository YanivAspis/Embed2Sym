import tensorflow as tf

LOSSES_DICT = {
    "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy,
    "binary_crossentropy": tf.keras.losses.BinaryCrossentropy
}

def get_loss_function(loss_name):
    return LOSSES_DICT[loss_name]