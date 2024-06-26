import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
width = 2560
height = 1440

calibrationImage = np.zeros((height, width,3), dtype= np.uint8)
calibrationImage = cv2.rectangle(calibrationImage, (20,20), (width-20, height-20), (0,255,0), 20)

# cv2.imshow("Calibration Image", calibrationImage)
# cv2.waitKey(0)

cv2.namedWindow("Calibration Frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Calibration Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Calibration Frame", calibrationImage)
cv2.waitKey(0)
time.sleep(3)
success, image = cap.read()

cv2.imshow("Captured Image", image)
cv2.imwrite('calibrate/images/CalibrationFrame.jpg', image)
cv2.waitKey(0)