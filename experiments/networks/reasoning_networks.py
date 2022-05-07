import tensorflow as tf

def mlp_reasoning(inputs, model_inputs = None):
    if len(inputs) == 1:
        inputs_concatenated = inputs[0]
    else:
        inputs_concatenated = tf.keras.layers.Concatenate()(inputs)
    output = tf.keras.Sequential([
            tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu)
    ])(inputs_concatenated)
    if model_inputs is None:
        return tf.keras.Model(inputs=tuple(inputs), outputs=output)
    else:
        return tf.keras.Model(inputs=tuple(model_inputs), outputs=output)

def mlp_reasoning_cifar10(inputs, model_inputs = None):
    if len(inputs) == 1:
        inputs_concatenated = inputs[0]
    else:
        inputs_concatenated = tf.keras.layers.Concatenate()(inputs)
    output = tf.keras.Sequential([
            tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])(inputs_concatenated)
    if model_inputs is None:
        return tf.keras.Model(inputs=tuple(inputs), outputs=output)
    else:
        return tf.keras.Model(inputs=tuple(model_inputs), outputs=output)



def member_attention_reasoning(inputs):
    embedded_symbolic_input = tf.keras.Sequential([
        tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu)
    ])(inputs[-1])
    attention = tf.keras.Sequential([
        tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu)
    ])
    attended_input = []
    for i in range(len(inputs)-1):
        attended_input.append(attention(tf.keras.layers.Concatenate()([inputs[i], embedded_symbolic_input])))
    attended_input_concatenated = tf.keras.layers.Concatenate()(attended_input)
    outputs = tf.keras.Sequential([
        tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu)
    ])(attended_input_concatenated)
    return tf.keras.Model(inputs=tuple(inputs), outputs=outputs)


def sort_reasoning(inputs, length):
    num_comaprisons = length * (length - 1) // 2
    comparison_inputs = tf.keras.layers.Concatenate()(inputs[:num_comaprisons])
    list_inputs = tf.stack(inputs[num_comaprisons:], axis=1)

    permutation_flattened = tf.keras.Sequential([
        tf.keras.layers.Dense(units=10 * length, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(units=10 * length, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(units=length * length, activation=tf.keras.activations.relu),
    ])(comparison_inputs)
    permutation = tf.keras.layers.Reshape((length, length))(permutation_flattened)
    output = tf.linalg.matmul(permutation, list_inputs)
    output = tf.unstack(output, axis=1)
    return tf.keras.Model(inputs=tuple(inputs), outputs=output)