import numpy as np
import cv2 as cv
import pickle

with open('./calibration.pckl', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    cameraMatrix, distCoeffs, rvecs, tvecs = pickle.load(f)


# img = cv.imread('/home/pi/opencv-examples/CalibrationByChessboard/camera-pic-of-chessboard-01.jpg')
img = cv.imread('./CalibrationByChessboard/test1296x972.jpg')
h, w, c = img.shape
# newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, cameraMatrix, distCoeffs, None, cameraMatrix)
 
# crop the image
x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
cv.imwrite('test1296x972_undistorted.png', dst)