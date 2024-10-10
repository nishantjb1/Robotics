import cv2
import numpy as np

# Load the image
img = cv2.imread('showbox_image.jpg')

# Check if the image is loaded correctly
if img is None:
    print("Error: Image not found or unable to load.")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help with edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply simple thresholding instead of adaptive thresholding
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set a lower minimum contour area and increase based on image resolution
    min_contour_area = 10000  # Adjust this value depending on your image size

    box_contour = None
    largest_area = 0  # Track the largest rectangular contour area
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > min_contour_area:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Ensure the contour has four vertices (likely rectangular)
            if len(approx) == 4:
                # Compute bounding box properties
                rect = cv2.minAreaRect(approx)
                width, height = rect[1]
                aspect_ratio = width / height if width > height else height / width

                # Only consider contours with a reasonable aspect ratio (rectangular)
                if 0.8 < aspect_ratio < 1.2:  # Approximate square aspect ratio
                    if area > largest_area:  # Update if it's the largest rectangular area
                        largest_area = area
                        box_contour = approx

    if box_contour is not None:
        # Get the minimum area rectangle around the contour
        rect = cv2.minAreaRect(box_contour)

        # Get the box points (corner points of the bounding rectangle)
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points)

        # Draw the bounding box (red) on the original image
        cv2.drawContours(img, [box_points], 0, (0, 0, 255), 2)

        # Draw a black square (the sticker) on the top face of the box
        sticker_size = 50  # Adjust this size based on the box size
        sticker_points = np.array([
            [-sticker_size // 2, -sticker_size // 2],
            [sticker_size // 2, -sticker_size // 2],
            [sticker_size // 2, sticker_size // 2],
            [-sticker_size // 2, sticker_size // 2]
        ])

        # Get the center and angle of the bounding box
        center = (int(rect[0][0]), int(rect[0][1]))
        angle = rect[2]

        # Create a rotation matrix to rotate the sticker according to the box orientation
        rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)

        # Apply the rotation matrix to the sticker points
        sticker_points_rotated = np.dot(rotation_matrix[:, :2], sticker_points.T).T
        sticker_points_rotated += center

        # Convert to int and draw the sticker
        sticker_points_rotated = np.int0(sticker_points_rotated)
        cv2.drawContours(img, [sticker_points_rotated], 0, (0, 0, 0), -1)  # Black filled square

        # Save the output image
        output_path = 'output1.jpg'
        cv2.imwrite(output_path, img)

        # Display the image
        cv2.namedWindow('Box with Sticker', cv2.WINDOW_NORMAL)
        cv2.imshow('Box with Sticker', img)

        print(f'Box center: {center}, Angle: {angle} degrees')
        print(f'Output image saved as: {output_path}')

        # Wait for a key press and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No suitable box contours found.")
