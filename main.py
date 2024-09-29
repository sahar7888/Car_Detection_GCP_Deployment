# Import libraries
from PIL import Image
import cv2
import numpy as np
import requests


image_path = "Data/cars.png"

img_car = cv2.imread(image_path)

img_car_gray = cv2.cvtColor(img_car, cv2.COLOR_BGR2GRAY )

# cv2.imshow('gray_car_img', img_car_gray)
#
# # Step 3: Wait for a key press
# cv2.waitKey(0)
#
# # Step 4: Destroy the window
# cv2.destroyAllWindows()

""" Apply Gaussian filter"""

blur = cv2.GaussianBlur(img_car_gray, (5,5), 0)

"""Dilation"""

dilated = cv2.dilate(blur, np.ones((3,3)))

"""Morphology"""
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)


"""Car Detection Model"""
car_cascade_src = 'cars.xml'

car_cascade = cv2.CascadeClassifier(car_cascade_src)

cars = car_cascade.detectMultiScale(closing, 1.1, 1)

"""Creating The Bounding Boxes"""

cnt = 0
for (x,y,w,h) in cars:
  cv2.rectangle(img_car,(x,y),(x+w,y+h),(255,0,0),2)
  cnt += 1

# Display the final image with rectangles
cv2.imshow('Detected Cars', img_car)

# Wait for a key press to close the window
cv2.waitKey(0)

# Destroy all OpenCV windows
cv2.destroyAllWindows()

# Print the count of detected cars (optional)
print(f"Number of detected cars: {cnt}")

