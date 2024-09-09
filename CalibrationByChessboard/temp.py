import cv2
import numpy as np

# Load the image
img = cv2.imread('CalibrationByChessboard/circleGrid_UwdCam_BW.png')

# Shrink the image by 50%
img = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a SimpleBlobDetector with custom parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 20
params.maxArea = 18000
params.minDistBetweenBlobs = 20
params.filterByColor = True
params.filterByConvexity = False
params.filterByCircularity = False
params.filterByInertia = False

detector = cv2.SimpleBlobDetector_create(params)

# Detect keypoints
keypoints = detector.detect(gray)

# Draw keypoints on the image
img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Mouse callback function
sorted_keypoints = []
def select_keypoint(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        min_dist = float('inf')
        selected_kp = None
        for kp in keypoints:
            dist = np.sqrt((kp.pt[0] - x) ** 2 + (kp.pt[1] - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                selected_kp = kp
        
        if selected_kp is not None:
            sorted_keypoints.append(selected_kp)
            print(f"Selected keypoint at: ({selected_kp.pt[0]}, {selected_kp.pt[1]})")
            cv2.circle(img_with_keypoints, (int(selected_kp.pt[0]), int(selected_kp.pt[1])), 15, (0, 255, 0), -1)
            
            # Print the number in red
            cv2.putText(img_with_keypoints, str(len(sorted_keypoints)), (int(selected_kp.pt[0]), int(selected_kp.pt[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow("Select Keypoints", img_with_keypoints)

# Display the image and set the mouse callback
cv2.namedWindow("Select Keypoints")
cv2.setMouseCallback("Select Keypoints", select_keypoint)
cv2.imshow("Select Keypoints", img_with_keypoints)

# Wait for user to finish selecting points
print("Click on the keypoints in the desired order. Press 'q' when done.")
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Print the sorted keypoints
sorted_points = np.array([kp.pt for kp in sorted_keypoints], dtype=np.float32)
print("Sorted keypoints:", sorted_points)

cv2.destroyAllWindows()

# Now `sorted_points` can be used for further processing
