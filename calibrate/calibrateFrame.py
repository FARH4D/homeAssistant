import cv2
import numpy as np

width = 2560
height = 1440

image = cv2.imread("calibrate/images/CalibrationFrame.jpg") # Gets the image of the frame on my desk
cv2.imshow("Image", image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresholdValue = 195

empty, thresholdedImage = cv2.threshold(gray, thresholdValue, 255, cv2.THRESH_BINARY)

thresholdedImage = thresholdedImage.astype('uint8')

# cv2.imshow("Thresholded Image", thresholdedImage)
# cv2.waitKey(0)

contours, empty = cv2.findContours(thresholdedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finds every contour in the image
largest = max(contours, key=cv2.contourArea) # Gets the largest contour (calculated based on the area)

cv2.drawContours(image, [largest], -1, (0, 0, 255), 2) # Draws the largest contour

# cv2.imshow("Largest Contour", image)
# cv2.waitKey(0)

epsilon = 0.02 * cv2.arcLength(largest, True) # Gets the perimeter of my frame
projectionCorners = cv2.approxPolyDP(largest, epsilon, True) # Creates a simple shape out of the outline of my frame (simplifies it if some of the lines are bent)
                                                            # It then gives me the 4 corners
if (len(projectionCorners) == 4):
    for i, corner in enumerate(projectionCorners): # Prints the coordinates of the corners of my frame
        x,y = corner.ravel()
        print(f"Corner {i + 1}: (x = {x}, y = {y})")
else:
    print("Not a quadrilateral.")
    

if (len(projectionCorners) == 4):
    points = projectionCorners.reshape((4, 2)).astype(np.float32) # Flattens the points array and converts it to float 32
    pointsSorted = points[np.argsort(points[:,0]), :] # Sort the points according to their x coordinates, left to right.
    
    topPoints = pointsSorted[:2, :]   # Separates the points into top and bottom points using the y coordinates.
    bottomPoints = pointsSorted[2:, :]
    
    topPoints = topPoints[np.argsort(topPoints[:, 1]), :] # Top points sorted from left to right
    bottomPoints = bottomPoints[np.argsort(bottomPoints[:, 1]), :] # Bottom points sorted from left to right
    
    orderedPoints = np.vstack([topPoints, bottomPoints[::-1]])
    
    dstPoints = np.array([ # Defines destination points
        [0, 0],
        [0, height - 1],
        [width - 1, height - 1],
        [width - 1, 0]
    ], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(orderedPoints, dstPoints)
    
    warpedImage = cv2.warpPerspective(image, M, (width, height))

    cv2.imshow("Final Image", warpedImage)
    cv2.waitKey(0)
    
    np.save("calibrate/M.npy", M)