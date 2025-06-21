import tensorflow as tf
from tensorflow.keras import layers, Model, Input

def build_model(input_shape):
    inputs = Input(shape=(input_shape,))
    
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1)(x)  # Regression output
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_absolute_error",
        metrics=["mae", "mse"]
    )
    
    return model
