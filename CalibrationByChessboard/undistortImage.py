import numpy as np
import cv2 as cv
import pickle

# Load calibration data from file
with open('calibrationHdUwdCam2.pckl', 'rb') as f:
    cameraMatrix, distCoeffs, rvecs, tvecs = pickle.load(f)

# Load the image to be undistorted
img = cv.imread('CalibrationByChessboard/test_image_2028x1520.png')
h, w, c = img.shape

# Define a new camera matrix (for example purposes, modify as needed)
new_cameraMatrix = np.array([
    [1400, 0, w / 2 - 100],  # Adjusted fx and cx
    [0, 1400, h / 2],        # Adjusted fy and cy
    [0, 0, 1]
], dtype=np.float32)

# Undistort the image using the new camera matrix
undistorted_img = cv.undistort(img, cameraMatrix, distCoeffs, None, new_cameraMatrix)

# Define source and destination points for perspective correction
src_points = np.float32([
    [0, 0],                # Top-left corner of the image
    [w - 1, 0],            # Top-right corner of the image
    [w - 1, h - 1],        # Bottom-right corner of the image
    [0, h - 1]             # Bottom-left corner of the image
])

# Define the destination points
dst_points = np.float32([
    [0, 0],                # Adjust according to the desired output (example values)
    [w - 1, 0],            # Top-right corner should remain the same
    [w - 1, h - 1],        # Bottom-right corner should remain the same
    [0, h - 1]             # Bottom-left corner should remain the same
])

# Compute the perspective transformation matrix
perspective_matrix = cv.getPerspectiveTransform(src_points, dst_points)

# Apply the perspective transformation
warped_img = cv.warpPerspective(undistorted_img, perspective_matrix, (w, h))

# Save the final undistorted and perspective-corrected image to a file
cv.imwrite('CalibrationByChessboard/test_image_2028x1520_corrected.png', warped_img)
