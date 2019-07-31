from tensorflow.python.keras import Model, layers, Input
from tensorflow.python.keras.models import model_from_config, Model
import tensorflow as tf
input = Input(shape=(10,))
h = layers.Dense(10)(input)
h = layers.Dense(11)(h)
out = layers.Dense(1)(h)
model = Model(inputs=input, outputs=out, name='test_model')


def clone_model(model):
    # Requires Keras 1.0.7 since get_config has breaking changes.
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config)
    clone.set_weights(model.get_weights())
    return clone

cloned_model = clone_model(model)

