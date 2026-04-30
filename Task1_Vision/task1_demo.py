import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse

TEXT_COLOR = (0, 255, 0)
POINT_COLOR = (0, 255, 255)
LINE_COLOR = (255, 0, 255)

def draw_landmarks_manual(image, landmarks, connections=None):
    """Draw landmarks and connections manually using OpenCV."""
    h, w = image.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    if connections:
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(image, points[start_idx], points[end_idx], LINE_COLOR, 1)
    for pt in points:
        cv2.circle(image, pt, 2, POINT_COLOR, -1)
    return image

# --------------------------------------------------------------------------- #
# 1. OBJECT DETECTION
# --------------------------------------------------------------------------- #
def object_detection(image_path):
    print("Running Object Detection...")
    base_options = python.BaseOptions(model_asset_path='models/efficientdet_lite0.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)

    image = mp.Image.create_from_file(image_path)
    result = detector.detect(image)

    img = np.copy(image.numpy_view())
    for det in result.detections:
        bbox = det.bounding_box
        p1 = (bbox.origin_x, bbox.origin_y)
        p2 = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(img, p1, p2, TEXT_COLOR, 2)
        label = f"{det.categories[0].category_name} {det.categories[0].score:.2f}"
        cv2.putText(img, label, (bbox.origin_x, bbox.origin_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    return img

# --------------------------------------------------------------------------- #
# 2. IMAGE CLASSIFICATION
# --------------------------------------------------------------------------- #
def image_classification(image_path):
    print("Running Image Classification...")
    base_options = python.BaseOptions(model_asset_path='models/efficientnet_lite0.tflite')
    options = vision.ImageClassifierOptions(base_options=base_options, max_results=3)
    classifier = vision.ImageClassifier.create_from_options(options)

    image = mp.Image.create_from_file(image_path)
    result = classifier.classify(image)

    img = np.copy(image.numpy_view())
    y = 35
    for cat in result.classifications[0].categories:
        text = f"{cat.category_name}: {cat.score:.2f}"
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
        y += 30
    return img

# --------------------------------------------------------------------------- #
# 3. IMAGE SEGMENTATION
# --------------------------------------------------------------------------- #
def image_segmentation(image_path):
    print("Running Image Segmentation...")
    base_options = python.BaseOptions(model_asset_path='models/deeplab_v3.tflite')
    options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
    segmenter = vision.ImageSegmenter.create_from_options(options)

    image = mp.Image.create_from_file(image_path)
    result = segmenter.segment(image)

    mask = result.category_mask.numpy_view()
    fg = (mask > 0).astype(np.uint8) * 255
    colored = cv2.applyColorMap(fg, cv2.COLORMAP_JET)

    img = image.numpy_view().copy()
    blended = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
    return blended

# --------------------------------------------------------------------------- #
# 4. HAND LANDMARK DETECTION
# --------------------------------------------------------------------------- #
# Simplified hand connections (tip → base of each finger, palm outline)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def hand_landmarks(image_path):
    print("Running Hand Landmark Detection...")
    base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    image = mp.Image.create_from_file(image_path)
    result = detector.detect(image)

    img = np.copy(image.numpy_view())
    for hand in result.hand_landmarks:
        draw_landmarks_manual(img, hand, HAND_CONNECTIONS)
    return img

# --------------------------------------------------------------------------- #
# 5. FACE LANDMARK DETECTION
# --------------------------------------------------------------------------- #
# A subset of face mesh connections for a visible result
FACE_OUTLINE = [
    (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),
    (356,454),(454,323),(323,361),(361,288),(288,397),(397,365),(365,379),
    (379,378),(378,400),(400,377),(377,152),(152,148),(148,176),(176,149),
    (149,150),(150,136),(136,172),(172,58),(58,132),(132,93),(93,234),
    (234,127),(127,162),(162,21),(21,54),(54,103),(103,67),(67,109),(109,10)
]

def face_landmarks(image_path):
    print("Running Face Landmark Detection...")
    base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    image = mp.Image.create_from_file(image_path)
    result = detector.detect(image)

    img = np.copy(image.numpy_view())
    for face in result.face_landmarks:
        # Draw all landmark points
        draw_landmarks_manual(img, face, FACE_OUTLINE)
    return img

# --------------------------------------------------------------------------- #
# 6. POSE LANDMARK DETECTION
# --------------------------------------------------------------------------- #
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32)
]

def pose_landmarks(image_path):
    print("Running Pose Landmark Detection...")
    base_options = python.BaseOptions(model_asset_path='models/pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    detector = vision.PoseLandmarker.create_from_options(options)

    image = mp.Image.create_from_file(image_path)
    result = detector.detect(image)

    img = np.copy(image.numpy_view())
    for pose in result.pose_landmarks:
        draw_landmarks_manual(img, pose, POSE_CONNECTIONS)
    return img

# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe Vision Tasks Demo")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--task", required=True,
                        choices=["object","classification","segmentation","hand","face","pose","all"],
                        help="Task to perform")
    args = parser.parse_args()

    tasks = {
        "object": object_detection,
        "classification": image_classification,
        "segmentation": image_segmentation,
        "hand": hand_landmarks,
        "face": face_landmarks,
        "pose": pose_landmarks,
    }

    run_tasks = list(tasks.items()) if args.task == "all" else [(args.task, tasks[args.task])]

    for name, func in run_tasks:
        result_img = func(args.image)
        display = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        window_title = f"MediaPipe - {name.upper()}"
        cv2.imshow(window_title, display)
        print(f"  [{name.upper()}] shown. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("Done!")
