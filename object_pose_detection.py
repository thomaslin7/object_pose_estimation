import cv2
import os # for loading file path
import numpy as np
import matplotlib.pyplot as plt # for plotting and data visualization
from sklearn.decomposition import PCA # for PCA, principal component analysis

# Function to detect if image is in this directory and return the image if true
def detect_image(image_name):
    # Check if image exists in the current directory
    if os.path.isfile(image_name):
        # Load and display image
        img = cv2.imread(image_name)
        return img
    else:
        print("Image Not Detected")
        return None

# Function to display the image
def display_image(img):
    # Display the image in a window
    cv2.imshow('Image', img)
    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Image filename
    image_name = 'dataset/level1/sanity_check.png'  # Change this to the correct image filename
    img = detect_image(image_name)

    # Convert the image to grayscale
    if img is not None:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply inverse binary thresholding --> black and white
    # _, binary = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY_INV) # second argument is the threshold value

    # Apply binary range thresholding
    binary = cv2.inRange(gray_img, 30, 200)

    # Apply inverse binary range thresholding
    # binary = cv2.inRange(gray_img, 0, 100) | cv2.inRange(gray_img, 200, 255)

    # Display the binary image
    display_image(binary)

    # Find nonzero points from the binary image
    points = np.column_stack(np.where(binary > 0))[:, ::-1] # [:, ::-1] flips the order of x and y

    # Apply PCA to find location and orientation
    pca = PCA(n_components=2) # number of principal components
    pca.fit(points)
    center = np.mean(points, axis=0) # get center coordinate of the object
    principal_axes = pca.components_ # get eigenvectors
    eigenvalues = pca.explained_variance_ # get eigenvalues

    # Visualize the center point
    cv2.circle(img, tuple(map(int, center)), 5, (255, 0, 0), -1)  # blue dot at center

    # Visualize principal axes
    colors = [(0, 0, 255), (0, 255, 0)] # color red for the first axis (long) and green for the second axis (short)
    for i, axis in enumerate(principal_axes):
        length = int(eigenvalues[i] * 0.2) # scale eigenvalue for visibility
        end_point = center + axis * length
        end_point = tuple(map(int, end_point))
        cv2.line(img, tuple(map(int, center)), end_point, colors[i], 2) # image / start point / end point / color / thickness

    # Display the image
    display_image(img)
