import cv2
import numpy as np

# Load image and convert to grayscale
img = cv2.imread('CalibrationByChessboard/circleGrid_UwdCam_BW.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

pattern_size = (14, 9)  # Number of circles per row and column
square_size = 0.290  # Change this to your actual square size in meters or any unit you use

# Create a SimpleBlobDetector with custom parameters
params = cv2.SimpleBlobDetector_Params()

# Filter by Area
params.filterByArea = True
params.minArea = 40
params.maxArea = 18000

params.minDistBetweenBlobs = 20

params.filterByColor = True
params.filterByConvexity = False

# Filter by Circularity
params.filterByCircularity = False

# Filter by Inertia
params.filterByInertia = False

# Create the detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect keypoints
keypoints = detector.detect(gray)

# Convert keypoints to a numpy array of (x, y) points
points = np.array([kp.pt for kp in keypoints], dtype=np.float32)

# Sort keypoints function
def sort_keypoints(points, pattern_size):
    # Sort by x-coordinate, then by y-coordinate
    sorted_by_x = points[np.argsort(points[:, 0])]
    
    # Split sorted points into rows
    rows = []
    num_points_per_row = pattern_size[0]
    
    for i in range(pattern_size[1]):
        row_points = sorted_by_x[i*num_points_per_row:(i+1)*num_points_per_row]
        rows.append(row_points)
    
    # Further sort each row by y-coordinate
    for i in range(pattern_size[1]):
        rows[i] = rows[i][np.argsort(rows[i][:, 1])]
    
    # Flatten the sorted rows into a single array
    sorted_points = np.vstack(rows)
    
    return sorted_points

# Sort keypoints using the improved function
sorted_keypoints = sort_keypoints(points, pattern_size)

# Create a blank image to draw the chessboard
chessboard_img = np.zeros_like(img)

# Draw keypoints on the image
for point in sorted_keypoints:
    x, y = point
    cv2.circle(chessboard_img, (int(x), int(y)), 10, (0, 255, 0), -1)

# Draw grid lines (for visualization of the chessboard pattern)
for i in range(pattern_size[1] + 1):
    cv2.line(chessboard_img, (0, i * (img.shape[0] // pattern_size[1])), 
             (img.shape[1], i * (img.shape[0] // pattern_size[1])), (255, 0, 0), 1)

for i in range(pattern_size[0] + 1):
    cv2.line(chessboard_img, (i * (img.shape[1] // pattern_size[0]), 0), 
             (i * (img.shape[1] // pattern_size[0]), img.shape[0]), (255, 0, 0), 1)

# Display the resulting image
cv2.imshow('Sorted Chessboard', chessboard_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
