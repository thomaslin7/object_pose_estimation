# Object Pose Estimation: Thresholding and PCA

A comprehensive educational resource on object pose estimation using image thresholding and principal component analysis (PCA). This repository is designed as part of a computer vision course, guiding students from simple thresholding operations to full object pose visualization and grasp planning.

## 🎯 Learning Objectives

This repository demonstrates:
- **Image Thresholding**: Techniques to separate objects from background (binary, inverse, range, HSV)
- **HSV Color Space**: Understanding hue, saturation, and value for robust color-based segmentation
- **Principal Component Analysis (PCA)**: Extracting object center and orientation from pixel data
- **Pose Visualization**: Drawing principal axes and parallel jaw grippers on objects
- **Scaling to Complexity**: Detecting single objects, multiple objects, and handling more challenging datasets

## 📚 Table of Contents

1. [Fundamentals](#-fundamentals)
2. [Thresholding Techniques](#-thresholding-techniques)
   - [Binary Thresholding](#binary-thresholding)
   - [Inverse Binary Thresholding](#inverse-binary-thresholding)
   - [Range Thresholding](#range-thresholding)
   - [HSV Color Space](#hsv-color-space)
3. [Pose Estimation with PCA](#-pose-estimation-with-pca)
   - [Single Object Pose](#single-object-pose)
   - [Parallel Jaw Gripper Visualization](#parallel-jaw-gripper-visualization)
   - [Multiple Object Pose Estimation](#multiple-object-pose-estimation)
4. [Running the Examples](#-running-the-examples)
5. [Homework](#-homework)

## 🔬 Fundamentals

### What is Object Pose Estimation?

Object pose estimation is the process of finding the position and orientation of an object in an image. This course uses a two-stage pipeline:

1. **Image Processing (Thresholding)**
   - Convert an image into a binary mask so the object can be isolated from the background.

2. **Principal Component Analysis (PCA)**
   - Apply PCA to the object pixels to compute its center and orientation.

Finally, the object's pose is visualized directly on the image with principal axes, and in advanced scripts, parallel jaw grippers are drawn to simulate grasp planning.

## 🎨 Thresholding Techniques

### Binary Thresholding

**File**: Included in `object_pose_detection.py`

**Purpose**: Separate bright objects from dark backgrounds using a single cutoff.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
```

**Example Use Case**: Detecting white objects on a black surface.

### Inverse Binary Thresholding

**File**: Included in `object_pose_detection.py`

**Purpose**: Detect darker objects on bright backgrounds.

```python
_, binary_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
```

**Example Use Case**: Black tools on a white workbench.

### Range Thresholding

**File**: Included in `object_pose_detection.py`

**Purpose**: Keep only pixel values within a range.

```python
binary_range = cv2.inRange(gray, 50, 200)
```

**Example Use Case**: Objects with medium brightness, while discarding very dark and very bright pixels.

### HSV Color Space

**File**: `HSV_color_space.py`

**Purpose**: Use hue, saturation, and value to filter objects based on color and brightness.

```python
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_hsv = np.array([0, 0, 150])
upper_hsv = np.array([180, 50, 255])
binary = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
```

**Example Use Case**: Detecting white packages on a conveyor belt with controlled lighting.

## 📐 Pose Estimation with PCA

### Single Object Pose

**File**: `object_pose_detection.py`

Applies PCA to a single object and visualizes:
- **Center** (blue dot)
- **Principal axes** (red = dominant axis, green = perpendicular axis)

### Parallel Jaw Gripper Visualization

**File**: `parallel_jaw_gripper.py`

“Extends single object PCA by adding parallel jaw gripper lines that grasp the object across its shorter axis.”

**Applications**: Robotics grasping simulation.

### Multiple Object Pose Estimation

**File**: `multiple_object_detection.py`

- Finds and filters multiple contours in an image.
- Applies PCA to each object individually.
- Visualizes axes, centers, and labels each object.

**Applications**: Detecting multiple packages on a conveyor belt.

## 🚀 Running the Examples

### Prerequisites

```bash
pip install opencv-python numpy matplotlib scikit-learn
```

### Dataset

The repository contains a `dataset/` folder with three difficulty levels:
- `level1/` → Easiest images, includes `sanity_check.png` for testing.
- `level2/` → Medium difficulty, more color/lighting variations.
- `level3/` → Hardest cases, realistic backgrounds and noise.

Each level contains 10 images.

### Execution

Run each script independently:

```bash
python object_pose_detection.py
python parallel_jaw_gripper.py
python multiple_object_detection.py
python HSV_color_space.py
```

Each script will display intermediate and final results in OpenCV windows.

## 📝 Homework

Design a student-driven mini-project to apply object pose estimation in real-world scenarios. Aim to finish in about 60 minutes.

### Instructions

- Use images from the provided `dataset/` (levels 1–3). Start with `level1/sanity_check.png`.
- You are also encouraged to take your own photos to test robustness.
- Experiment with different thresholding techniques: binary, inverse binary, range, and HSV.
- Apply PCA to detected objects and visualize the results with principal axes and centers.
- Work progressively:
  - Start with level1 images (easy).
  - Move to level2 and level3 to challenge yourself.

### Tasks

1. **Problem Statement & Data**
   - Pick 2–3 images from different levels (or your own).
   - Write 2–3 sentences describing the challenge (e.g., lighting variation, noisy background, low contrast).

2. **Thresholding Experiments**
   - Try at least two thresholding methods per image.
   - Save intermediate results (binary mask, contours).

3. **Pose Estimation with PCA**
   - Apply PCA and draw principal axes on each object.
   - For multi-object scenes, label each object (#1, #2, #3).
   - Save final PCA visualization results.

4. **Parameter Exploration**
   - Adjust thresholds and note how results change.
   - Record both good and bad cases, and explain why.

5. **Challenge Yourself**
   - Try at least one image from level2 and one from level3.
   - Reflect on what makes these harder (color, saturation, shadows, etc.).

6. **Reflection & Solutions**
   - Record challenges faced (e.g., noise, overlapping objects).
   - Propose possible solutions (e.g., filtering, better lighting, controlled backgrounds).

### Deliverables

- `results/` folder containing:
  - Binary masks and PCA visualizations for each chosen image.
- A short `HOMEWORK.md` with:
  - Parameters used (threshold values, HSV ranges).
  - 2–3 insights per experiment.
  - A reflection on what worked, what failed, and how you might improve it.

### Optional Extensions (if time permits)

- Capture your own real-world images and run the pipeline.
- Compare results on the same object under different lighting conditions.
- Experiment with environment control (e.g., solid backgrounds) to simplify thresholding.

---

**Happy Learning!** 🎯

*This repository serves as a foundation for understanding object pose estimation in computer vision and robotics applications.*
