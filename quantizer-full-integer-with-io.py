import tensorflow as tf
import numpy as np
import argparse
import os

# --- Parse CLI arguments ---
parser = argparse.ArgumentParser(description='Convert Keras model to fully quantized INT8 TFLite model.')
parser.add_argument('--mname', type=str, required=True, help='Name of the Keras model file (without .h5)')
parser.add_argument('--direct', type=str, default='models/', help='Directory containing the model')
args = parser.parse_args()

mname = args.mname
model_path = os.path.join(args.direct, mname + '.h5')

# --- Load Keras model ---
model = tf.keras.models.load_model(model_path, compile=False)

# --- Setup TFLite Converter ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Use only INT8 ops
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Force input and output tensors to INT8
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# --- Define representative dataset generator ---
def representative_data_gen():
    for _ in range(100):
        # Replace shape (64, 64, 1) with your model’s input shape if different
        dummy_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        yield [dummy_input]

converter.representative_dataset = representative_data_gen

# --- Convert model ---
tflite_model = converter.convert()

# --- Save to file ---
output_path = mname + '-int8.tflite'
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ INT8 TFLite model saved as: {output_path}")

