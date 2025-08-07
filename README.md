# Computer Vision Fundamentals: Object Detection & Pose Estimation

A comprehensive collection of computer vision techniques for object detection, pose estimation, and robotic manipulation. This repository serves as an educational resource for understanding fundamental computer vision concepts, particularly image processing, color space analysis, and geometric object analysis.

## 🎯 Learning Objectives

This repository demonstrates:
- **Image Thresholding**: Converting images to binary for object segmentation
- **HSV Color Space**: Understanding hue, saturation, and value for color-based detection
- **Object Detection**: Finding and analyzing multiple objects in images
- **Pose Estimation**: Determining object orientation using Principal Component Analysis (PCA)
- **Robotic Applications**: Visualizing gripper positioning for object manipulation

## 📚 Table of Contents

1. [Fundamentals](#-fundamentals)
2. [Core Techniques](#-core-techniques)
   - [Image Thresholding](#image-thresholding)
   - [HSV Color Space](#hsv-color-space)
   - [Object Detection](#object-detection)
   - [Pose Estimation](#pose-estimation)
   - [Multiple Object Analysis](#multiple-object-analysis)
   - [Robotic Gripper Visualization](#robotic-gripper-visualization)
3. [Running the Examples](#-running-the-examples)
4. [Educational Resources](#-educational-resources)

## 🔬 Fundamentals

### What is Computer Vision?

Computer vision is a field of artificial intelligence that enables computers to interpret and understand visual information from the world. This involves processing images to extract meaningful information such as object locations, orientations, and properties.

### Key Concepts Covered

1. **Image Processing**: Converting and manipulating images for analysis
2. **Color Spaces**: Understanding different ways to represent color information
3. **Object Segmentation**: Separating objects from background
4. **Geometric Analysis**: Understanding object shape and orientation
5. **Robotic Applications**: Applying computer vision to robotic manipulation

## 🎨 Core Techniques

### Image Thresholding

**Purpose**: Converting grayscale images to binary (black and white) for object segmentation.

**Types of Thresholding**:

1. **Binary Thresholding**: Simple threshold with two values
```python
# Binary thresholding
_, binary = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)

# Inverse binary thresholding
_, binary = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY_INV)
```

2. **Range Thresholding**: Thresholding within a specific range
```python
# Range thresholding
binary = cv2.inRange(gray_img, 30, 200)

# Inverse range thresholding
binary = cv2.inRange(gray_img, 0, 100) | cv2.inRange(gray_img, 200, 255)
```

**Applications**:
- Object segmentation
- Background removal
- Pre-processing for other operations

### HSV Color Space

**File**: `HSV_color_space.py`

**Purpose**: Using HSV (Hue, Saturation, Value) color space for more robust color-based object detection.

**HSV Components**:
- **Hue**: Color type (0-180 in OpenCV)
- **Saturation**: Color intensity (0-255)
- **Value**: Brightness (0-255)

```python
# Convert BGR to HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define HSV range for object detection
lower_hsv = np.array([0, 0, 150])    # Lower threshold
upper_hsv = np.array([180, 50, 255]) # Upper threshold

# Create binary mask
binary = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
```

**Advantages**:
- More robust to lighting changes
- Better color separation
- Easier to define color ranges

**Applications**:
- Color-based object detection
- Lighting-invariant segmentation
- Multi-color object identification

### Object Detection

**Purpose**: Finding and analyzing objects in images using contour detection.

**Process**:
1. **Binary Image Creation**: Convert to black and white
2. **Contour Detection**: Find object boundaries
3. **Noise Filtering**: Remove small contours
4. **Object Analysis**: Analyze each detected object

```python
# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Filter small contours (noise)
min_contour_area = 200
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Draw contours
cv2.drawContours(img, filtered_contours, -1, (0, 255, 0), 2)
```

**Applications**:
- Object counting
- Quality control
- Automated inspection

### Pose Estimation

**File**: `object_pose_detection.py`

**Purpose**: Determining object orientation using Principal Component Analysis (PCA).

**PCA Process**:
1. **Point Extraction**: Get object points from binary image
2. **PCA Analysis**: Find principal axes (eigenvectors)
3. **Orientation Calculation**: Determine object orientation
4. **Visualization**: Draw principal axes

```python
# Get object points
points = np.column_stack(np.where(binary > 0))[:, ::-1]

# Apply PCA
pca = PCA(n_components=2)
pca.fit(points)

# Get center and principal axes
center = np.mean(points, axis=0)
principal_axes = pca.components_

# Visualize
cv2.circle(img, tuple(map(int, center)), 5, (255, 0, 0), -1)
for i, axis in enumerate(principal_axes):
    length = int(eigenvalues[i] * 0.2)
    end_point = center + axis * length
    cv2.line(img, tuple(map(int, center)), end_point, colors[i], 2)
```

**Applications**:
- Robotic manipulation
- Object alignment
- Quality inspection

### Multiple Object Analysis

**File**: `multiple_object_detection.py`

**Purpose**: Analyzing multiple objects simultaneously with individual pose estimation.

**Process**:
1. **Multi-Object Detection**: Find all objects in image
2. **Individual Analysis**: Apply PCA to each object
3. **Object Labeling**: Number and identify each object
4. **Comprehensive Visualization**: Show all objects with their properties

```python
# Process each contour individually
for contour in filtered_contours:
    points = np.squeeze(contour)
    
    # Skip if insufficient points
    if points.ndim == 1 or points.shape[0] < 3:
        continue
    
    # Apply PCA to each object
    pca = PCA(n_components=2)
    pca.fit(points)
    
    # Calculate object properties
    center = np.mean(points, axis=0)
    principal_axes = pca.components_
    
    # Visualize each object
    cv2.circle(img, tuple(map(int, center)), 5, (255, 0, 0), -1)
    cv2.putText(img, f"#{count}", tuple(map(int, center + [10, 10])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
```

**Applications**:
- Multi-object sorting
- Assembly line inspection
- Robotic pick-and-place

### Robotic Gripper Visualization

**File**: `parallel_jaw_gripper.py`

**Purpose**: Visualizing how a parallel jaw gripper would approach and grasp objects.

**Gripper Design**:
- **Parallel Jaws**: Two parallel lines for grasping
- **Orientation-Based**: Aligned with object's principal axes
- **Safety Offset**: Distance from object to prevent collision

```python
# Calculate gripper position
gripper_offset = 30  # Distance from object
half_gripper_length = 100

# Draw parallel gripper jaws
for sign in [-1, 1]:  # Two parallel lines
    start_point = center + sign * principal_axes[1] * (principal_axes_lengths[1] + gripper_offset) + principal_axes[0] * half_gripper_length
    end_point = center + sign * principal_axes[1] * (principal_axes_lengths[1] + gripper_offset) - principal_axes[0] * half_gripper_length
    cv2.line(img, start_point, end_point, (0, 165, 255), 2)
```

**Applications**:
- Robotic manipulation planning
- Grasp pose optimization
- Industrial automation

## 🚀 Running the Examples

### Prerequisites

```bash
pip install opencv-python numpy matplotlib scikit-learn
```

### Execution

Each technique can be run independently:

```bash
# Basic pose estimation
python object_pose_detection.py

# HSV color space analysis
python HSV_color_space.py

# Multiple object detection
python multiple_object_detection.py

# Robotic gripper visualization
python parallel_jaw_gripper.py
```

### Expected Output

Each script will display:
- **Original Image**: The input image
- **Binary Image**: Thresholded result
- **Analysis Results**: Object detection and pose estimation
- **Visualization**: Annotated image with detected objects and orientations

## 📖 Educational Resources

### Key Concepts Covered

1. **Image Processing Pipeline**: From raw image to object analysis
2. **Color Space Understanding**: BGR vs HSV representation
3. **Mathematical Foundations**: PCA for orientation analysis
4. **Robotic Applications**: Real-world computer vision applications

### Practical Exercises

1. **Threshold Tuning**: Adjust threshold values for different images
2. **HSV Range Experimentation**: Modify HSV ranges for different objects
3. **Multi-Object Scenarios**: Test with images containing multiple objects
4. **Gripper Parameter Adjustment**: Modify gripper dimensions and offsets

---

**Happy Learning!** 🤖

*This repository serves as a foundation for understanding computer vision fundamentals in robotics and automation applications.*
