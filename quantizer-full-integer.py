import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import keras

mname = 'shallow_wide'

model = tf.keras.models.load_model(mname + '.h5', compile = False)
#model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Use TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable post-training quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Set representative dataset function (to calibrate quantization)
def representative_data_gen():
    for _ in range(100):
        # Replace with your actual input data
        yield [np.random.rand(1, 64, 64, 1).astype(np.float32)]

converter.representative_dataset = representative_data_gen

# Convert and save the quantized model
tflite_model = converter.convert()

#keras.utils.plot_model(model, to_file="bigger_mobilenet.svg", show_shapes=True)

with open(mname + '.tflite', 'wb') as f:
    f.write(tflite_model)
