import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse

# Colors for drawing
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

def object_detection(image_path):
    print("Running Object Detection...")
    base_options = python.BaseOptions(model_asset_path='models/efficientdet_lite0.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)
    
    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)
    
    # Draw bounding boxes
    image_copy = np.copy(image.numpy_view())
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(image_copy, start_point, end_point, TEXT_COLOR, 3)
        
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f'{category_name} ({probability})'
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image_copy, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    
    return image_copy

def image_classification(image_path):
    print("Running Image Classification...")
    base_options = python.BaseOptions(model_asset_path='models/efficientnet_lite0.tflite')
    options = vision.ImageClassifierOptions(base_options=base_options, max_results=3)
    classifier = vision.ImageClassifier.create_from_options(options)
    
    image = mp.Image.create_from_file(image_path)
    classification_result = classifier.classify(image)
    
    image_copy = np.copy(image.numpy_view())
    y_offset = 30
    for category in classification_result.classifications[0].categories:
        text = f"{category.category_name}: {category.score:.2f}"
        cv2.putText(image_copy, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)
        y_offset += 30
        
    return image_copy

def image_segmentation(image_path):
    print("Running Image Segmentation...")
    base_options = python.BaseOptions(model_asset_path='models/deeplab_v3.tflite')
    options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
    segmenter = vision.ImageSegmenter.create_from_options(options)
    
    image = mp.Image.create_from_file(image_path)
    segmentation_result = segmenter.segment(image)
    
    category_mask = segmentation_result.category_mask.numpy_view()
    
    # Create an overlaid image (e.g. coloring the background or foreground)
    # The mask contains class indices.
    fg_mask = (category_mask > 0).astype(np.uint8) * 255
    colored_mask = cv2.applyColorMap(fg_mask, cv2.COLORMAP_JET)
    
    # Blend
    image_copy = image.numpy_view()
    alpha = 0.5
    blended = cv2.addWeighted(image_copy, alpha, colored_mask, 1 - alpha, 0)
    return blended

def hand_landmarks(image_path):
    print("Running Hand Landmark Detection...")
    base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    
    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)
    
    image_copy = np.copy(image.numpy_view())
    from mediapipe.framework.formats import landmark_pb2
    for hand_landmarks in detection_result.hand_landmarks:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            image_copy,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())
    return image_copy

def face_landmarks(image_path):
    print("Running Face Landmark Detection...")
    base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    
    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)
    
    image_copy = np.copy(image.numpy_view())
    from mediapipe.framework.formats import landmark_pb2
    for face_landmarks in detection_result.face_landmarks:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            image=image_copy,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
    return image_copy

def pose_landmarks(image_path):
    print("Running Pose Landmark Detection...")
    base_options = python.BaseOptions(model_asset_path='models/pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    detector = vision.PoseLandmarker.create_from_options(options)
    
    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)
    
    image_copy = np.copy(image.numpy_view())
    from mediapipe.framework.formats import landmark_pb2
    for pose_landmarks in detection_result.pose_landmarks:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            image_copy,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    return image_copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe Vision Tasks Demo")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--task", required=True, choices=["object", "classification", "segmentation", "hand", "face", "pose", "all"], help="Task to perform")
    
    args = parser.parse_args()
    
    tasks = {
        "object": object_detection,
        "classification": image_classification,
        "segmentation": image_segmentation,
        "hand": hand_landmarks,
        "face": face_landmarks,
        "pose": pose_landmarks
    }
    
    if args.task == "all":
        for t_name, t_func in tasks.items():
            res = t_func(args.image)
            cv2.imshow(t_name, cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        res = tasks[args.task](args.image)
        cv2.imshow(f"MediaPipe - {args.task}", cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
