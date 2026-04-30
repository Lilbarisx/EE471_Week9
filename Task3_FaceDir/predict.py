import cv2
import mediapipe as mp
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5
        )

    def predict(
        self,
        image: Path = Input(description="Input image for face direction classification")
    ) -> str:
        """Run a single prediction on the model"""
        image_cv = cv2.imread(str(image))
        if image_cv is None:
            return "Error: Could not read image."
            
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return "None (No face detected)"
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        # We can use the nose tip (index 1), left cheek edge (index 234) and right cheek edge (index 454)
        nose_x = landmarks[1].x
        left_edge_x = landmarks[234].x
        right_edge_x = landmarks[454].x
        
        # Ensure left_edge is actually visually left. In image coordinates, x increases to the right.
        # But face left and right might be mirrored. 
        # Actually, index 234 is the left side of the face (from the person's perspective), which is on the right side of the image if facing the camera.
        # Let's just use min_x and max_x of the face.
        min_x = min(left_edge_x, right_edge_x)
        max_x = max(left_edge_x, right_edge_x)
        
        face_width = max_x - min_x
        # Ratio of nose position relative to the face width
        # 0.0 means nose is at the extreme left (looking right from our perspective)
        # 1.0 means nose is at the extreme right (looking left from our perspective)
        if face_width == 0:
            return "straight"
            
        ratio = (nose_x - min_x) / face_width
        
        # Typical straight faces have ratio around 0.5. Let's use 0.4 to 0.6 as straight.
        if ratio < 0.4:
            return "right"  # Nose is pointing towards the left of the image
        elif ratio > 0.6:
            return "left"   # Nose is pointing towards the right of the image
        else:
            return "straight"
