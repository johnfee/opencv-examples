import pickle
import configparser
import numpy as np

# Load calibration parameters
with open('./calibrationWdCam2.pckl', 'rb') as f:
    cameraMatrix, distCoeffs, rvecs, tvecs = pickle.load(f)

# Create a ConfigParser object
config = configparser.ConfigParser()

# Add a section
config.add_section('CameraCalibration')

# Convert numpy arrays to strings and add them as key-value pairs to the section
config.set('CameraCalibration', 'cameraMatrix', np.array2string(cameraMatrix, separator=','))

# Flatten the distCoeffs array to avoid double brackets if it's a single row
distCoeffs_flat = distCoeffs.flatten() if distCoeffs.shape[0] == 1 else distCoeffs
config.set('CameraCalibration', 'distCoeffs', np.array2string(distCoeffs_flat, separator=','))

# Convert each array inside rvecs and tvecs to a string and store them as a single string
rvecs_str = ','.join([np.array2string(rvec, separator=',') for rvec in rvecs])
tvecs_str = ','.join([np.array2string(tvec, separator=',') for tvec in tvecs])

config.set('CameraCalibration', 'rvecs', rvecs_str)
config.set('CameraCalibration', 'tvecs', tvecs_str)

# Write the configuration to an ini file
with open('calibrationWdCam2.ini', 'w') as configfile:
    config.write(configfile)

print('Data written to calibrationWdCam2.ini')
