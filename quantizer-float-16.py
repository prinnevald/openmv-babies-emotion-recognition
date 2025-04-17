import tensorflow as tf
import numpy as np
import argparse

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description='Convert a Keras model to TFLite with float16 quantization.')
parser.add_argument('--mname', type=str, required=True, help='Model name without extension')
parser.add_argument('--direct', type=str, default='models/', help='Directory where model is stored (default: models/)')
args = parser.parse_args()

# Load the Keras model
model = tf.keras.models.load_model(args.direct + args.mname + '.h5', compile=False)

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set float16 support
converter.target_spec.supported_types = [tf.float16]

# Convert model
tflite_model = converter.convert()

# Save model
with open(args.mname + '-float-16.tflite', 'wb') as f:
    f.write(tflite_model)

