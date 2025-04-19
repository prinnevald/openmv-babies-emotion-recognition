# Facial Expression Recognition
#
# This project uses a pre-trained Facial Expression Recognition TFlite model.


import sensor
import image
import time
import tf

sensor.reset()
# Set grayscale with 320x240 size
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QVGA)

LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

print(dir(tf))

net = tf.load("model_quantized.tflite", load_to_fb=True)
face_cascade = image.HaarCascade("frontalface", stages=25)
print("Loaded model: %s and a face detector" % (net))

clock = time.clock()

while(True):
    clock.tick()

    # Take a picture and brighten things up for the frontal face detector.
    img = sensor.snapshot().gamma_corr(contrast=1.5)

    objects = img.find_features(face_cascade, threshold=0.75, scale_factor=1.25)

    # Classify the Image
    for obj in objects:
        scores = net.classify(img, roi=obj)[0].output()
        max_idx = scores.index(max(scores))
        print("Emotion: %s = %f" % (LABELS[max_idx], scores[max_idx]))
        img.to_rgb565()
        img.draw_rectangle(obj, color=(255,0,0))
        img.draw_string(obj[0] + 3, obj[1] - 1, LABELS[max_idx], mono_space=False, color=(255,0,0))


    # Print FPS.
    # Note: Actual FPS is higher, streaming the FB makes it slower.
    print(clock.fps())
