import os
import urllib.request

MODELS = {
    "efficientdet_lite0.tflite": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
    "efficientnet_lite0.tflite": "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite",
    "deeplab_v3.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
}

# Official MediaPipe sample images - one per task
SAMPLE_IMAGES = {
    "samples/object_detection.jpg": "https://storage.googleapis.com/mediapipe-tasks/object_detector/cat_and_dog.jpg",
    "samples/classification.jpg":   "https://storage.googleapis.com/mediapipe-tasks/image_classifier/burger.jpg",
    "samples/segmentation.jpg":     "https://storage.googleapis.com/mediapipe-assets/portrait.jpg",
    "samples/hand.jpg":             "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg",
    "samples/face.jpg":             "https://storage.googleapis.com/mediapipe-assets/portrait.jpg",
    "samples/pose.jpg":             "https://storage.googleapis.com/mediapipe-assets/image.jpg",
}

os.makedirs("models", exist_ok=True)
os.makedirs("samples", exist_ok=True)

print("=== Downloading models ===")
for name, url in MODELS.items():
    path = os.path.join("models", name)
    if not os.path.exists(path):
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(url, path)
        print(f"  [OK] {name}")
    else:
        print(f"  [OK] {name} (already exists)")

print("\n=== Downloading sample images ===")
for path, url in SAMPLE_IMAGES.items():
    if not os.path.exists(path):
        print(f"  Downloading {path}...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  [OK] {path}")
        except Exception as e:
            print(f"  [FAIL] Could not download {path}: {e}")
    else:
        print(f"  [OK] {path} (already exists)")

print("\nAll done! Ready to run task1_demo.py")
