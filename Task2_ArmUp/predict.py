import cv2
import mediapipe as mp
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True, 
            min_detection_confidence=0.5
        )

    def predict(
        self,
        image: Path = Input(description="Input image for pose estimation")
    ) -> str:
        """Run a single prediction on the model"""
        image_cv = cv2.imread(str(image))
        if image_cv is None:
            return "Error: Could not read image."
        
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return "None (No pose detected)"
            
        landmarks = results.pose_landmarks.landmark
        
        # Get shoulder and wrist landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # In image coordinates, y=0 is at the top. 
        # A smaller y value means the point is higher up.
        # We consider an arm "up" if the wrist is significantly higher than the shoulder.
        left_arm_up = left_wrist.y < left_shoulder.y
        right_arm_up = right_wrist.y < right_shoulder.y
        
        if left_arm_up and right_arm_up:
            return "both"
        elif left_arm_up:
            return "left"
        elif right_arm_up:
            return "right"
        else:
            return "None"
