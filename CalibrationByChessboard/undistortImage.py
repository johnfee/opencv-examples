import numpy as np
import cv2 as cv
import pickle

with open('calibrationHdUwdCam.pckl', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    cameraMatrix, distCoeffs, rvecs, tvecs = pickle.load(f)


# img = cv.imread('/home/pi/opencv-examples/CalibrationByChessboard/camera-pic-of-chessboard-01.jpg')
img = cv.imread('CalibrationByChessboard/circleGrid_HqUwdCam_2028x1520.png')
h, w, c = img.shape
# newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, cameraMatrix, distCoeffs, None, cameraMatrix)
 
# crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
cv.imwrite('CalibrationByChessboard/circleGrid_HqUwdCam_2028x1520_undistorted.png', dst)