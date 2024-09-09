import cv2
import numpy as np

# Load image and convert to grayscale
img = cv2.imread('CalibrationByChessboard/circleGrid_HqUwdCam_2028x1520_BW.png')
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

# Draw keypoints on the image for visual reference
img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Scale down the image for display
scale_factor = 0.5  # Adjust this factor as needed to fit your screen
img_display = cv2.resize(img_with_keypoints, None, fx=scale_factor, fy=scale_factor)

# Mouse callback function for selecting keypoints manually
sorted_keypoints = []

def select_keypoint(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Map the clicked point back to the original image size
        orig_x, orig_y = int(x / scale_factor), int(y / scale_factor)
        
        min_dist = float('inf')
        selected_kp = None
        for kp in keypoints:
            dist = np.sqrt((kp.pt[0] - orig_x) ** 2 + (kp.pt[1] - orig_y) ** 2)
            if dist < min_dist:
                min_dist = dist
                selected_kp = kp
        
        if selected_kp is not None:
            sorted_keypoints.append(selected_kp)
            print(f"Selected keypoint at: ({selected_kp.pt[0]}, {selected_kp.pt[1]})")
            cv2.circle(img_display, (int(selected_kp.pt[0] * scale_factor), int(selected_kp.pt[1] * scale_factor)), 
                       15, (0, 255, 0), -1)
            
            # Print the number in red
            cv2.putText(img_display, str(len(sorted_keypoints)), 
                        (int(selected_kp.pt[0] * scale_factor), int(selected_kp.pt[1] * scale_factor)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Update the window
            cv2.imshow("Select Keypoints", img_display)

# Display the image and set the mouse callback
cv2.namedWindow("Select Keypoints", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Select Keypoints", select_keypoint)

# Show the image
cv2.imshow("Select Keypoints", img_display)

# Wait for user to finish selecting points
print("Click on the keypoints in the desired order. Press 'q' when done.")
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Convert selected keypoints to numpy array
sorted_points = np.array([kp.pt for kp in sorted_keypoints], dtype=np.float32)

# Create object points (3D points in the world space)
object_points = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
object_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
object_points *= square_size

# Prepare image points
img_points = [sorted_points.reshape(-1, 2)]

# Add the object points and image points to lists
obj_points = [object_points]

# Convert image points to the correct format
img_points = [np.array(p, dtype=np.float32) for p in img_points]

# Use cv2.calibrateCamera
flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_RATIONAL_MODEL
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None, flags=flags
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

cv2.destroyAllWindows()
