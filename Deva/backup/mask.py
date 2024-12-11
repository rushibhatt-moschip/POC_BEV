import numpy as np
import matplotlib.pyplot as plt
import cv2

# Function to compute the slope between two points
def compute_slope(x1, y1, x2, y2):
    if x2 - x1 == 0:
        raise ValueError("The x-values cannot be the same, as it would result in division by zero.")
    return (y2 - y1) / (x2 - x1)

# Function to determine if a point (x, y) is above or below the line
def point_relative_to_line(x, y, x1, y1, slope):
    # Equation of the line: y = slope * (x - x1) + y1
    line_y = slope * (x - x1) + y1
    return y < line_y  # True if the point is above the line

# Define the two points (x1, y1) and (x2, y2)
x1, y1 = 780, 90  # First point
x2, y2 = 580, 505  # Second point

canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Compute the slope of the line
slope = compute_slope(x1, y1, x2, y2)

for i in range(720):
    for j in range(1280):
        if point_relative_to_line(j, i, x1, y1, slope):  
            canvas[i, j] = [255, 255, 255]  

cv2.imwrite("/tmp/mask_right_00.jpg",canvas)

