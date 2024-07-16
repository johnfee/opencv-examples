import cv2
import cv2.aruco as aruco

# Create ChArUco board, which is a set of Aruco markers in a chessboard setting
# meant for calibration
# the following call gets a ChArUco board of tiles 5 wide X 7 tall
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
gridboard = cv2.aruco.CharucoBoard((11, 8), 0.015, 0.011, aruco_dict)

# Create an image from the gridboard
img = gridboard.draw(outSize=(988, 1400))
cv2.imwrite("test_charuco.jpg", img)

# Display the image to us
cv2.imshow('Gridboard', img)
# Exit on any key
cv2.waitKey(0)
cv2.destroyAllWindows()

