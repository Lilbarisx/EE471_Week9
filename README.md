# EE471 Week 9 In-Class Exercise

This repository contains the implementation for the Week 9 in-class exercises, utilizing MediaPipe for various vision tasks and Cog/Docker for containerized model prediction.

## Repository Structure

- `Task1_Vision/`: Demonstrations of 6 MediaPipe vision tasks (Object Detection, Image Classification, Image Segmentation, Hand Landmarks, Face Landmarks, Pose Landmarks).
- `Task2_ArmUp/`: A Cog-containerized pose estimation script that classifies which arm is up.
- `Task3_FaceDir/`: A Cog-containerized face detection script that classifies the looking direction (left, right, or straight).

---

## 🚀 Task 1: Vision Tasks Demonstrations

To run Task 1, you need to set up a Python environment and download the necessary MediaPipe model weights.

### Setup

1. Open a terminal in the `Task1_Vision` directory.
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   py -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install mediapipe opencv-python matplotlib
   ```
4. Download the required model bundles:
   ```bash
   py download_models.py
   ```

### Execution

You can run the script on any image. Use `--task` to choose a specific task (`object`, `classification`, `segmentation`, `hand`, `face`, `pose`) or `all` to run all 6 at once.

```bash
# Example usage:
py task1_demo.py --image path/to/your/image.jpg --task all
```
*(If you need a quick test image, you can copy one of the `pose-1.jpg` or `face-1.png` files from the other folders).*

---

## 🐳 Task 2: Arm Position Classification

This task is containerized using [Cog](https://github.com/replicate/cog). It takes an image as input and outputs `"left"`, `"right"`, `"both"`, or `"None"` depending on which arm is raised.

### Execution

1. Open a terminal (WSL2 or Linux/macOS with Docker running) in the `Task2_ArmUp` directory.
2. Test the script with the provided images using `cog predict`:

```bash
cog predict -i image=@pose-1.jpg
cog predict -i image=@pose-2.jpg
cog predict -i image=@pose-3.jpg
```

---

## 🐳 Task 3: Face Looking Direction

This task is also containerized using Cog. It classifies whether a face is looking `"left"`, `"right"`, or `"straight"`.

### Execution

1. Open a terminal (WSL2 or Linux/macOS with Docker running) in the `Task3_FaceDir` directory.
2. Test the script with the provided images using `cog predict`:

```bash
cog predict -i image=@face-1.png
cog predict -i image=@face-2.png
cog predict -i image=@face-3.png
```

---

## 🎥 Deliverables Guide
- **Code:** You can push this directory to your GitHub repository.
- **Videos:** 
  1. Record a video demonstrating Task 1 (`py task1_demo.py --image ... --task all`).
  2. Record a video demonstrating Task 2 and Task 3 running via `cog predict`.
  3. Record an explanation video where you go through the Python scripts (`task1_demo.py`, `predict.py` in Task 2 & 3) and explain the logic used (e.g., comparing wrist vs shoulder y-coordinates for Task 2, comparing nose tip position relative to face width for Task 3).
