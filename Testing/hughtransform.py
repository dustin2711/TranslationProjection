import cv2
import numpy as np
import math


def compute_skew(filename):
    # Load image in grayscale
    src = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Invert colors (background black, text white)
    src = cv2.bitwise_not(src)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(src, 50, 150, apertureSize=3)

    # Probabilistic Hough Transform to detect lines
    min_line_length = src.shape[1] // 2  # Minimum line length
    max_line_gap = 20  # Maximum allowed gap between points on the same line
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=min_line_length, maxLineGap=max_line_gap
    )

    # Display detected lines on the image for visualization
    disp_lines = np.zeros_like(src)
    angle = 0.0
    nb_lines = len(lines)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(disp_lines, (x1, y1), (x2, y2), 255, 1)
        angle += math.atan2(y2 - y1, x2 - x1)  # Calculate angle of each line

    # Calculate the mean angle (in radians)
    mean_angle = angle / nb_lines
    angle_degrees = mean_angle * 180 / np.pi  # Convert to degrees

    print(f"Detected skew angle: {angle_degrees:.2f} degrees")

    # Display results
    cv2.imshow("Detected Lines", disp_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Run the function on an example image
compute_skew("japanese.png")
