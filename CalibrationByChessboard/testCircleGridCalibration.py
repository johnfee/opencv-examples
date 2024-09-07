import cv2
import numpy as np

# Load image and convert to grayscale
img = cv2.imread('CalibrationByChessboard/circleGrid_UwdCam_BW.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

pattern_size = (14, 9)  # Number of circles per row and column
square_size = 0.025  # Change this to your actual square size in meters or any unit you use

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

# Create object points (3D points in the world space)
object_points = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
object_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
object_points *= square_size

# Prepare image points
img_points = [sorted_keypoints.reshape(-1, 2)]

# Add the object points and image points to lists
obj_points = [object_points]

# Convert image points to the correct format
img_points = [np.array(p, dtype=np.float32) for p in img_points]

# Use cv2.calibrateCamera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

# Check if calibration was successful
if ret:
    print("Calibration successful!")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
else:
    print("Calibration failed.")


# Undistort the grid
undistorted_grid = cv2.undistort(img, camera_matrix, dist_coeffs)

# Display the undistorted grid
cv2.imshow('Undistorted Grid', undistorted_grid)
cv2.waitKey(0)

# Save the undistorted grid
cv2.imwrite('undistorted_grid.png', undistorted_grid)


# Undistort the image
foto = cv2.imread('CalibrationByChessboard/TableWdCam2_undistorted.png')
undistorted_img = cv2.undistort(foto, camera_matrix, dist_coeffs)

# Display the undistorted image
cv2.imshow('Undistorted Foto', undistorted_img)
cv2.waitKey(0)

# Save the undistorted image
cv2.imwrite('undistorted_foto2.png', undistorted_img)

# Create a copy of the original image to draw on
im_with_keypoints = img.copy()

# Draw keypoints and their numbers
for idx, kp in enumerate(sorted_keypoints.reshape(-1, 2)):
    position = (int(kp[0]), int(kp[1]))
    
    # Draw a black circle for the keypoint
    cv2.circle(im_with_keypoints, position, 15, (0, 0, 0), -1)  # Larger circle for better visibility
    
    # Add a background rectangle behind the text
    text = str(idx)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = position[0] - text_size[0] // 2
    text_y = position[1] + text_size[1] // 2
    
    # Draw a filled rectangle for the text background
    cv2.rectangle(im_with_keypoints, (text_x - 5, text_y - text_size[1] - 5), 
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    
    # Draw the text in white
    cv2.putText(im_with_keypoints, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# Display the image with keypoints and their numbers
cv2.imshow('Sorted Keypoints with Numbers', im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
