import tensorflow as tf

METRICS_DICT = {
    "categorical_accuracy": tf.keras.metrics.CategoricalAccuracy,
    "binary_accuracy": tf.keras.metrics.BinaryAccuracy
}

def get_metric_function(metric_name):
    return METRICS_DICT[metric_name]