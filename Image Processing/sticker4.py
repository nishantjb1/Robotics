import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('showbox1.jpg')
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

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area and get the largest ones
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

# Find the contour with 4 vertices (assuming it's the top face of the box)
top_face = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:
        top_face = approx
        break

if top_face is not None:
    # Reshape the top face contour
    top_face = top_face.reshape(-1, 2)
    
    # Calculate center
    center = np.mean(top_face, axis=0).astype(int)
    
    # Calculate orientation
    edge = top_face[1] - top_face[0]
    angle = np.arctan2(edge[1], edge[0]) * 180 / np.pi
    
    # Adjust angle to be relative to the horizontal
    if angle < 0:
        angle += 180
    
    # Draw results
    result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, [top_face], 0, (0, 255, 0), 2)
    cv2.circle(result, tuple(center), 5, (0, 0, 255), -1)
    
    print(f"Center coordinates: {center}")
    print(f"Angle with horizontal: {angle:.2f} degrees")
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Detected Top Face')
    plt.axis('off')
    plt.show()
else:
    print("Could not detect a quadrilateral top face.")