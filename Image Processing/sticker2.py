import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('showbox2.jpg')
if image is None:
    raise ValueError("Image not found or couldn't be loaded")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection
edges = cv2.Canny(blurred, 50, 150)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(132), plt.imshow(gray, cmap='gray'), plt.title('Grayscale')
plt.subplot(133), plt.imshow(edges, cmap='gray'), plt.title('Edges')
plt.tight_layout()
plt.show()



# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area and get the largest one (assuming it's the box)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
box_contour = contours[0]

# Draw the contour on the original image
result = image.copy()
cv2.drawContours(result, [box_contour], 0, (0, 255, 0), 2)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Detected Box')
plt.axis('off')
plt.show()