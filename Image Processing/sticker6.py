import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv

def isolate_box(hsv_image):
    # Define range of box color in HSV
    # H: 0-179, S: 0-255, V: 0-255
    lower_color = np.array([13, 75, 163])  # Lower bound for yellow-ish color
    upper_color = np.array([179, 231, 255])  # Upper bound for yellow-ish color
    
    # Create trackbars for color range adjustment
    cv2.namedWindow('Color Range Adjustment')
    cv2.createTrackbar('H Lower', 'Color Range Adjustment', lower_color[0], 179, lambda x: None)
    cv2.createTrackbar('S Lower', 'Color Range Adjustment', lower_color[1], 255, lambda x: None)
    cv2.createTrackbar('V Lower', 'Color Range Adjustment', lower_color[2], 255, lambda x: None)
    cv2.createTrackbar('H Upper', 'Color Range Adjustment', upper_color[0], 179, lambda x: None)
    cv2.createTrackbar('S Upper', 'Color Range Adjustment', upper_color[1], 255, lambda x: None)
    cv2.createTrackbar('V Upper', 'Color Range Adjustment', upper_color[2], 255, lambda x: None)

    while True:
        # Get current positions of trackbars
        h_lower = cv2.getTrackbarPos('H Lower', 'Color Range Adjustment')
        s_lower = cv2.getTrackbarPos('S Lower', 'Color Range Adjustment')
        v_lower = cv2.getTrackbarPos('V Lower', 'Color Range Adjustment')
        h_upper = cv2.getTrackbarPos('H Upper', 'Color Range Adjustment')
        s_upper = cv2.getTrackbarPos('S Upper', 'Color Range Adjustment')
        v_upper = cv2.getTrackbarPos('V Upper', 'Color Range Adjustment')

        # Update color range
        lower_color = np.array([h_lower, s_lower, v_lower])
        upper_color = np.array([h_upper, s_upper, v_upper])

        # Threshold the HSV image to get only box colors
        mask = cv2.inRange(hsv_image, lower_color, upper_color)
    
        # Bitwise-AND mask and original image
        box_only = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
    
        # Display the result
        cv2.imshow('Isolated Box', cv2.cvtColor(box_only, cv2.COLOR_HSV2BGR))
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return box_only, mask


def detect_edges(image, mask):
    # Convert box_only to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Use the mask to keep only edges of the box
    edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    return edges


def find_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    
    if lines is None:
        return []
    
    return lines


def filter_horizontal_vertical_lines(lines, angle_threshold=10):
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
        if abs(angle) < angle_threshold or abs(angle - 180) < angle_threshold:
            horizontal_lines.append(line)
        elif abs(angle - 90) < angle_threshold or abs(angle + 90) < angle_threshold:
            vertical_lines.append(line)
    return horizontal_lines, vertical_lines

def find_intersections(horizontal_lines, vertical_lines):
    intersections = []
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            x1, y1, x2, y2 = h_line[0]
            x3, y3, x4, y4 = v_line[0]
            
            # Calculate intersection point
            denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
            if denom != 0:  # Avoid division by zero
                px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
                py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
                intersections.append((int(px), int(py)))
    return intersections

def find_top_face_corners(intersections):
    # Assuming the top face is the largest quadrilateral
    if len(intersections) < 4:
        return None
    
    # Find convex hull of intersections
    hull = cv2.convexHull(np.array(intersections))
    hull = [point[0] for point in hull]
    
    # Find the 4 corners that form the largest area
    max_area = 0
    top_face_corners = None
    for i in range(len(hull)):
        for j in range(i+1, len(hull)):
            for k in range(j+1, len(hull)):
                for l in range(k+1, len(hull)):
                    quad = np.array([hull[i], hull[j], hull[k], hull[l]], dtype=np.float32)
                    area = cv2.contourArea(quad)
                    if area > max_area:
                        max_area = area
                        top_face_corners = quad
    
    return top_face_corners

def calculate_center_and_orientation(corners):
    # Calculate center
    center = np.mean(corners, axis=0)
    
    # Calculate orientation (angle of longest edge)
    edges = []
    for i in range(4):
        edge = corners[i] - corners[(i+1)%4]
        edges.append((np.linalg.norm(edge), edge))
    
    longest_edge = max(edges, key=lambda x: x[0])[1]
    angle = math.atan2(longest_edge[1], longest_edge[0])
    
    return center, angle

def place_sticker(image, center, angle):
    sticker_size = 50  # Size of the sticker in pixels
    half_size = sticker_size // 2
    
    # Create a black square
    sticker = np.zeros((sticker_size, sticker_size, 3), dtype=np.uint8)
    
    # Create a rotation matrix
    M = cv2.getRotationMatrix2D((half_size, half_size), angle * 180 / np.pi, 1)
    
    # Rotate the sticker
    rotated_sticker = cv2.warpAffine(sticker, M, (sticker_size, sticker_size))
    
    # Calculate the position to place the sticker
    x = int(center[0] - half_size)
    y = int(center[1] - half_size)
    
    # Place the sticker on the image
    roi = image[y:y+sticker_size, x:x+sticker_size]
    mask = cv2.cvtColor(rotated_sticker, cv2.COLOR_BGR2GRAY)
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    sticker_fg = cv2.bitwise_and(rotated_sticker, rotated_sticker, mask=mask)
    dst = cv2.add(img_bg, sticker_fg)
    image[y:y+sticker_size, x:x+sticker_size] = dst
    
    return image

def visualize_step(image, title, output_folder):
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.tight_layout()
    plt.show()


def process_image(image_path, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    plt.figure(figsize=(15, 5))
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")
    
    visualize_step(image, "1_Original_Image", output_folder)
    
    # Preprocess the image
    hsv = preprocess_image(image)
    visualize_step(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "2_HSV_Image", output_folder)
    
    # Isolate the box
    box_only, mask = isolate_box(hsv)
    visualize_step(cv2.cvtColor(box_only, cv2.COLOR_HSV2BGR), "3_Isolated_Box", output_folder)
    visualize_step(mask, "3b_Box_Mask", output_folder)
    
    # Detect edges
    edges = detect_edges(box_only, mask)
    visualize_step(edges, "4_Edge_Detection", output_folder)
    
    # Find lines
    lines = find_lines(edges)
    if not lines:
        raise ValueError("No lines detected in the image")
    
    # Visualize all detected lines
    line_image = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    visualize_step(line_image, "5_All_Detected_Lines", output_folder)
    
    # Filter horizontal and vertical lines
    horizontal_lines, vertical_lines = filter_horizontal_vertical_lines(lines)
    
    # Visualize filtered lines
    filtered_line_image = image.copy()
    for line in horizontal_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(filtered_line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for line in vertical_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(filtered_line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    visualize_step(filtered_line_image, "6_Filtered_Lines", output_folder)
    
    # Find intersections
    intersections = find_intersections(horizontal_lines, vertical_lines)
    
    # Visualize intersections
    intersection_image = image.copy()
    for point in intersections:
        cv2.circle(intersection_image, point, 5, (0, 0, 255), -1)
    visualize_step(intersection_image, "7_Intersections", output_folder)
    
    # Find top face corners
    corners = find_top_face_corners(intersections)
    if corners is None:
        raise ValueError("Could not identify the top face of the box")
    
    # Visualize top face
    top_face_image = image.copy()
    cv2.drawContours(top_face_image, [np.int0(corners)], 0, (0, 255, 0), 2)
    visualize_step(top_face_image, "8_Top_Face", output_folder)
    
    # Calculate center and orientation
    center, angle = calculate_center_and_orientation(corners)
    
    # Place sticker
    result = place_sticker(image.copy(), center, angle)
    visualize_step(result, "9_Final_Result", output_folder)
    
    return result

# Main execution
if __name__ == "__main__":
    image_path = "showbox6.jpg"  # Replace with your image path
    output_folder = "output_images"  # Folder to save intermediate results
    try:
        result = process_image(image_path, output_folder)
        print(f"Processing complete. Check the '{output_folder}' directory for step-by-step results.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")