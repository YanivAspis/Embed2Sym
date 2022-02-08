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
