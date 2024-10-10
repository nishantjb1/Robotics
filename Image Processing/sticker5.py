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

contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Function to check if a contour is likely to be the top face
def is_top_face(contour, image_shape):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h
    area = cv2.contourArea(contour)
    # Adjust these criteria based on your specific image
    if 0.5 < aspect_ratio < 2 and area > 10000 and y < image_shape[0]/2:
        return True
    return False

# Find the contour that's most likely to be the top face
top_face = None
max_area = 0
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) == 4 and is_top_face(approx, edges.shape):
        area = cv2.contourArea(approx)
        if area > max_area:
            top_face = approx
            max_area = area

if top_face is not None:
    # Reshape the top face contour
    top_face = top_face.reshape(-1, 2)
    
    # Calculate center
    center = np.mean(top_face, axis=0).astype(int)
    
    # Calculate orientation
    edge = top_face[1] - top_face[0]
    angle = np.arctan2(edge[1], edge[0]) * 180 / np.pi
    
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
    print("Could not detect the top face of the box.")

# def intersection(line1, line2):
#     rho1, theta1 = line1
#     rho2, theta2 = line2
#     A = np.array([
#         [np.cos(theta1), np.sin(theta1)],
#         [np.cos(theta2), np.sin(theta2)]
#     ])
#     b = np.array([[rho1], [rho2]])
#     x0, y0 = np.linalg.solve(A, b)
#     return int(np.round(x0)), int(np.round(y0))

# lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

# # Filter lines to get only those likely to be part of the top face
# top_lines = []
# for line in lines:
#     rho, theta = line[0]
#     if (np.pi/4 < theta < np.pi*3/4) or (np.pi*5/4 < theta < np.pi*7/4):
#         top_lines.append((rho, theta))

# # Find intersections
# corners = []
# for i in range(len(top_lines)):
#     for j in range(i+1, len(top_lines)):
#         point = intersection(top_lines[i], top_lines[j])
#         if 0 <= point[0] < edges.shape[1] and 0 <= point[1] < edges.shape[0]:
#             corners.append(point)

# # Get the 4 corners that form the largest quadrilateral
# corners = np.array(corners)
# hull = cv2.convexHull(corners)
# top_face = cv2.approxPolyDP(hull, 10, True)

# if len(top_face) == 4:
#     top_face = np.array(top_face).reshape(-1, 2)
    
#     # Calculate center
#     center = np.mean(top_face, axis=0).astype(int)
    
#     # Calculate orientation
#     edge = top_face[1] - top_face[0]
#     angle = np.arctan2(edge[1], edge[0]) * 180 / np.pi
    
#     # Draw results
#     result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#     cv2.drawContours(result, [top_face], 0, (0, 255, 0), 2)
#     cv2.circle(result, tuple(center), 5, (0, 0, 255), -1)
    
#     print(f"Center coordinates: {center}")
#     print(f"Angle with horizontal: {angle:.2f} degrees")
    
#     plt.figure(figsize=(10, 10))
#     plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
#     plt.title('Detected Top Face')
#     plt.axis('off')
#     plt.show()
# else:
#     print("Could not detect a quadrilateral top face.")


# contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# # Function to check if a contour is likely to be the top face
# def is_top_face(contour, image_shape):
#     x, y, w, h = cv2.boundingRect(contour)
#     aspect_ratio = float(w)/h
#     area = cv2.contourArea(contour)
#     if 0.5 < aspect_ratio < 2 and area > 1000 and y < image_shape[0]/2:
#         return True
#     return False

# # Find the contour that's most likely to be the top face
# top_face = None
# for contour in contours:
#     approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
#     if len(approx) == 4 and is_top_face(approx, edges.shape):
#         top_face = approx
#         break

# if top_face is not None:
#     # Reshape the top face contour
#     top_face = top_face.reshape(-1, 2)
    
#     # Calculate center
#     center = np.mean(top_face, axis=0).astype(int)
    
#     # Calculate orientation
#     edge = top_face[1] - top_face[0]
#     angle = np.arctan2(edge[1], edge[0]) * 180 / np.pi
    
#     # Adjust angle to be relative to the horizontal
#     if angle < 0:
#         angle += 180
    
#     # Draw results
#     result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#     cv2.drawContours(result, [top_face], 0, (0, 255, 0), 2)
#     cv2.circle(result, tuple(center), 5, (0, 0, 255), -1)
    
#     print(f"Center coordinates: {center}")
#     print(f"Angle with horizontal: {angle:.2f} degrees")
    
#     plt.figure(figsize=(10, 10))
#     plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
#     plt.title('Detected Top Face')
#     plt.axis('off')
#     plt.show()
# else:
#     print("Could not detect the top face of the box.")