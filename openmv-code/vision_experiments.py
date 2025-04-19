import sensor, image, tf, os

model_name = "model_quantized"
labels = ["angry", "fear", "happy", "sad", "surprise", "neutral"]
label_to_index = {label: i for i, label in enumerate(labels)}

net = tf.load(model_name + ".tflite", load_to_fb=True)
print("Model loaded.")

sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)

total = 0
correct = 0

for label in labels:
    folder_path = "test/" + label
    try:
        files = os.listdir(folder_path)
        for filename in files:
            img_path = folder_path + "/" + filename
            img = image.Image(img_path, copy_to_fb=True)

            scores = net.classify(img)[0].output()
            pred_idx = scores.index(max(scores))

            if label_to_index[label] == pred_idx:
                correct += 1
            total += 1
    except Exception as e:
        print("Skipping folder:", label, "-", e)

# Save accuracy to a file on the SD card
if total > 0:
    accuracy = correct / total
    result_text = "{:.4f}".format(accuracy)
    try:
        with open("/sd/" + model_name + "-results.txt", "w") as f:
            f.write(result_text)
        print("Saved to SD:", result_text)
    except Exception as e:
        print("Error saving to SD card:", e)
else:
    print("No test images found.")
