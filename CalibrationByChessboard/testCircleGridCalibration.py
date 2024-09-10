import cv2
import numpy as np
import os

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

# File path to store the sorted keypoints
sorted_keypoints_file = 'sorted_keypoints.npy'

# Function to manually select keypoints if no saved keypoints are found
def select_keypoints_manually():
    sorted_keypoints = []

    def select_keypoint(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
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

                cv2.putText(img_display, str(len(sorted_keypoints)),
                            (int(selected_kp.pt[0] * scale_factor), int(selected_kp.pt[1] * scale_factor)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow("Select Keypoints", img_display)

    cv2.namedWindow("Select Keypoints", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Keypoints", select_keypoint)
    cv2.imshow("Select Keypoints", img_display)

    print("Click on the keypoints in the desired order. Press 'q' when done.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    sorted_points = np.array([kp.pt for kp in sorted_keypoints], dtype=np.float32)
    
    # Save sorted keypoints to a file
    np.save(sorted_keypoints_file, sorted_points)
    return sorted_points

# Load previously sorted keypoints if they exist
if os.path.exists(sorted_keypoints_file):
    print("Loading sorted keypoints from previous run.")
    sorted_points = np.load(sorted_keypoints_file)
else:
    print("No saved keypoints found. Please select the keypoints manually.")
    sorted_points = select_keypoints_manually()

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

# Create a white canvas for drawing the distorted chessboard
canvas = np.ones_like(img) * 255

# Draw the distorted chessboard pattern (middle)
cols, rows = pattern_size
square_count = 1
for r in range(rows - 1): 
    for c in range(cols - 1):
        idx1 = r * cols + c
        idx2 = idx1 + 1
        idx3 = (r + 1) * cols + c
        idx4 = idx3 + 1

        if square_count % 2 == 1 and idx1 < len(sorted_points) and idx2 < len(sorted_points) and idx3 < len(sorted_points) and idx4 < len(sorted_points):
            pts = np.array([sorted_points[idx1], sorted_points[idx2], sorted_points[idx4], sorted_points[idx3]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(canvas, [pts], color=(0, 0, 0))  # Fill the square
        square_count += 1
        
# Draw the distorted chessboard pattern (top line)
square_count = 1
for r in range(cols - 1): 
    id1 = r
    id2 = id1 + 1  
    x1=sorted_points[id1][0]
    y1=sorted_points[id1][1]  
    x2=sorted_points[id2][0]
    y2=sorted_points[id2][1] 
    
    if square_count % 2 != 1 and id1 < len(sorted_points) and id2 < len(sorted_points):
        # Define the points of the polygon
        pts = np.array([[x1, y1], [x2, y2], [x2, y2-50], [x1, y1-50], [x1, y1]], np.int32)
        pts = pts.reshape((-1, 1, 2))  # Required shape for fillPoly        
        cv2.fillPoly(canvas, [pts], color=(0, 0, 0))  # Fill the square
    square_count += 1
    
# Draw the distorted chessboard pattern (bottom line)
square_count = 1
for r in range(cols - 1): 
    id1 = rows*cols - cols + r
    id2 = id1 + 1  
    x1=sorted_points[id1][0]
    y1=sorted_points[id1][1]  
    x2=sorted_points[id2][0]
    y2=sorted_points[id2][1] 
    
    if square_count % 2 == 1 and id1 < len(sorted_points) and id2 < len(sorted_points):
        # Define the points of the polygon
        pts = np.array([[x1, y1], [x2, y2], [x2, y2+50], [x1, y1+50], [x1, y1]], np.int32)
        pts = pts.reshape((-1, 1, 2))  # Required shape for fillPoly        
        cv2.fillPoly(canvas, [pts], color=(0, 0, 0))  # Fill the square
    square_count += 1
    
# Draw the distorted chessboard pattern (left line)
square_count = 1
for r in range(rows - 1): 
    id1 = r*cols
    id2 = id1 + cols  
    x1=sorted_points[id1][0]
    y1=sorted_points[id1][1]  
    x2=sorted_points[id2][0]
    y2=sorted_points[id2][1] 
    
    if square_count % 2 != 1 and id1 < len(sorted_points) and id2 < len(sorted_points):
        # Define the points of the polygon
        pts = np.array([[x1, y1], [x2, y2], [x2-50, y2], [x1-50, y1], [x1, y1]], np.int32)
        pts = pts.reshape((-1, 1, 2))  # Required shape for fillPoly        
        cv2.fillPoly(canvas, [pts], color=(0, 0, 0))  # Fill the square
    square_count += 1

# Draw the distorted chessboard pattern (right line)
square_count = 1
for r in range(rows - 1): 
    id1 = r*cols + cols -1
    id2 = id1 + cols  
    x1=sorted_points[id1][0]
    y1=sorted_points[id1][1]  
    x2=sorted_points[id2][0]
    y2=sorted_points[id2][1] 
    
    if square_count % 2 != 1 and id1 < len(sorted_points) and id2 < len(sorted_points):
        # Define the points of the polygon
        pts = np.array([[x1, y1], [x2, y2], [x2+50, y2], [x1+50, y1], [x1, y1]], np.int32)
        pts = pts.reshape((-1, 1, 2))  # Required shape for fillPoly        
        cv2.fillPoly(canvas, [pts], color=(0, 0, 0))  # Fill the square
    square_count += 1
    
    # Draw the distorted chessboard pattern (top left square)
    x0=sorted_points[0][0]
    y0=sorted_points[0][1]
    
    pts = np.array([[x0, y0], [x0, y0-50], [x0-50, y0-50], [x0-50, y0], [x0-50, y0]], np.int32)
    pts = pts.reshape((-1, 1, 2))  # Required shape for fillPoly        
    cv2.fillPoly(canvas, [pts], color=(0, 0, 0))  # Fill the square
    
    # Draw the distorted chessboard pattern (top right square)
    x0=sorted_points[cols-1][0]
    y0=sorted_points[cols-1][1]
    
    pts = np.array([[x0, y0], [x0, y0-50], [x0+50, y0-50], [x0+50, y0], [x0+50, y0]], np.int32)
    pts = pts.reshape((-1, 1, 2))  # Required shape for fillPoly        
    cv2.fillPoly(canvas, [pts], color=(0, 0, 0))  # Fill the square
    


# Display the distorted chessboard pattern
cv2.imshow('Distorted Chessboard Pattern', canvas)
cv2.waitKey(0)

# Save the distorted chessboard pattern
cv2.imwrite('distorted_chessboard_pattern.png', canvas)

# Draw keypoints and their numbers
im_with_keypoints = img.copy()

# Draw keypoints and their numbers
for idx, kp in enumerate(sorted_points.reshape(-1, 2)):
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
    
    # Draw the text in red
    cv2.putText(im_with_keypoints, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

# Display the image with keypoints and their numbers
cv2.imshow('Sorted Keypoints with Numbers', im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
