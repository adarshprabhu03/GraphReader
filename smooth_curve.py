import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract
import math
from scipy.interpolate import UnivariateSpline

def find_crop_coordinates(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to handle uneven lighting
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Perform closing to fill small gaps in the lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours in the closed image
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box of the largest contour (assuming it's the line graph)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add some margin to the bounding box
        margin = 10
        x1, y1, x2, y2 = max(0, x - margin), max(0, y - margin), x + w + margin, y + h + margin
        return x1, y1, x2, y2

    # If no contours are found, return the entire image
    return 0, 0, image.shape[1], image.shape[0]

def crop_image(image, x1, y1, x2, y2):
    cropped_image = image[y1:y2, x1:x2]
    height, width, _ = cropped_image.shape
    print(height,width)
    return (image[y1:y2, x1:x2], height, width)

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Find dynamic crop coordinates
    x1, y1, x2, y2 = find_crop_coordinates(img)

    # Crop the image
    img_cropped, height, width = crop_image(img, x1, y1, x2, y2)

    # Convert the cropped image to grayscale
    gray_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve line detection
    blurred = cv2.GaussianBlur(gray_cropped, (5, 5), 0)

    return img, blurred, (x1, y1, x2, y2)

def detect_lines(blurred):
    # Apply Canny edge detection with lower and higher thresholds
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)

    # Use probabilistic Hough Transform to find lines with modified parameters
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=5)

    return lines

def filter_lines(lines, slope_threshold=0.2):
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate the slope of the line
        slope = (y2 - y1) / (x2 - x1 + 1e-5)  # Avoid division by zero

        # Filter out horizontal and vertical lines
        if abs(slope) > slope_threshold and abs(slope) < 1/slope_threshold:
            filtered_lines.append(line)

    return filtered_lines

def draw_detected_lines(image, lines, crop_coords):
    x1_crop, y1_crop, _, _ = crop_coords

    img_with_lines = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Draw lines in red (BGR format)
        cv2.line(img_with_lines, (x1 + x1_crop, y1 + y1_crop), (x2 + x1_crop, y2 + y1_crop), (0, 0, 255), 2)

    return img_with_lines

def extract_data_points(lines):
    data_points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        data_points.append((x1, y1))
        data_points.append((x2, y2))

    # Sort data points by x-coordinate in descending order
    return sorted(data_points, key=lambda x: x[0], reverse=True)

def filter_data_points(data_points, angle_threshold=20):
    filtered_data_points = []
    n = len(data_points)

    for i in range(n):
        # Get three consecutive points
        prev_point = data_points[i - 1]
        current_point = data_points[i]
        next_point = data_points[(i + 1) % n]  # Wrap around for the last point

        # Calculate vectors between points
        vector1 = np.array(prev_point) - np.array(current_point)
        vector2 = np.array(next_point) - np.array(current_point)

        # Calculate angle between vectors (in degrees)
        angle = np.degrees(np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-5)))

        # Filter out points based on the angle
        if angle > angle_threshold:
            filtered_data_points.append(current_point)

    return filtered_data_points

def plot_data_points(image, data_points, lines, crop_coords):
    df = pd.DataFrame(data_points, columns=['X', 'Y'])

    # Plot the image with detected lines
    img_with_lines = image.copy()
  

    # Display the image with detected lines
    plt.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
    
    # Extract the crop coordinates
    x1_crop, y1_crop, _, _ = crop_coords
    
    # Plot the filtered data points with proper overlay
    plt.scatter(df['X'] + x1_crop, df['Y'] + y1_crop, color='red', label='Filtered Data Points')
    plt.title('Image with Detected Lines and Filtered Data Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()

    return df

def extract_text_from_image(image, region):
    x1, y1, x2, y2 = region
    cropped_region = image[y1:y2, x1:x2]

    # Use Tesseract OCR to extract text
    text = pytesseract.image_to_string(cropped_region, config='--psm 6')

    # Clean and strip whitespace from the extracted text
    return text.strip().replace(' ', '')

def extract_x_y_values(image, crop_coords):
    x1, y1, x2, y2 = crop_coords
    x_axis_region = (x1, y2, x2, y2 + 50)
    y_axis_region = (x1 - 50, y1, x1, y2)

    # Extract text from the x-axis and y-axis regions
    x_axis_text = extract_text_from_image(image, x_axis_region)
    y_axis_text = extract_text_from_image(image, y_axis_region)

    # Handle potential errors in extracted text
    x_values = _parse_values(x_axis_text)
    y_values = _parse_values(y_axis_text)

    # Check if both axes have values
    if not x_values or not y_values:
        raise ValueError("Failed to extract x or y-axis values")

    return x_values, y_values

def _parse_values(text):
    values = []
    for line in text.splitlines():
        for value in line.split():
            try:
                values.append(float(value))
            except ValueError:
                # Ignore non-numerical values
                pass
    return values

def plot_original_graph(x_values, y_values, height, width, x_max, y_max, x_offset, y_offset):
    x_max_value = max(x_values)
    y_max_value = max(y_values)
    x_scale = width / x_max
    y_scale = height / y_max
    modified_x_values = [round((x / width) * x_max) + x_offset for x in x_values]
    modified_y_values = [(((height - y) / height) * y_max) + y_offset for y in y_values]

    # Add a small offset to make x values unique
    modified_x_values, unique_indices = np.unique(modified_x_values, return_index=True)
    modified_y_values = [modified_y_values[i] for i in unique_indices]

    # Use UnivariateSpline for smooth curve
    spline = UnivariateSpline(modified_x_values, modified_y_values, s=0)

    # Generate smooth curve points
    x_smooth = np.linspace(min(modified_x_values), max(modified_x_values), 1000)
    y_smooth = spline(x_smooth)

    # Create a DataFrame with modified x and y values
    df_modified = pd.DataFrame({'Modified_X': modified_x_values, 'Modified_Y': modified_y_values})

    # Print the DataFrame
    print("Modified Dataframe:")
    print(df_modified)

    # Plot the smooth curve
    plt.plot(x_smooth, y_smooth, color='blue', label='Smooth Curve')
    plt.scatter(modified_x_values, modified_y_values, marker='o', color='red', label='Data Points')
    plt.title('Original Graph with Smooth Curve')
    plt.grid(True)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()

def main():
    image_path = '/image/path'

    # Preprocess the image and get dynamic crop coordinates
    original_image, blurred_image, crop_coords = preprocess_image(image_path)

    # Display the original image
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.show()

    # Display the cropped image
    img_cropped, height, width = crop_image(original_image, *crop_coords)
    plt.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Image')
    plt.show()

    # Detect lines in the cropped image
    detected_lines = detect_lines(blurred_image)

    # Draw detected lines on the cropped image
    img_with_lines = draw_detected_lines(original_image, detected_lines, crop_coords)
    plt.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Image with Detected Lines')
    plt.show()

    # Filter lines based on slope
    filtered_lines = filter_lines(detected_lines)

    # Draw filtered lines on the cropped image
    img_with_filtered_lines = draw_detected_lines(original_image, filtered_lines, crop_coords)
    plt.imshow(cv2.cvtColor(img_with_filtered_lines, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Image with Filtered Lines')
    plt.show()

    # Extract filtered data points from filtered lines
    filtered_data_points = filter_data_points(extract_data_points(filtered_lines))

    # Plot the data points for verification
    df_filtered = plot_data_points(original_image, filtered_data_points, filtered_lines, crop_coords)
    print("Filtered DataFrame:")
    #print(df_filtered)

    # Extract filtered x-axis and y-axis values using OCR
    
    x_values_filtered = df_filtered['X'].tolist()
    y_values_filtered = df_filtered['Y'].tolist()

    # Plot the original graph using the extracted filtered x-axis and y-axis values
    if x_values_filtered and y_values_filtered:
        plot_original_graph(x_values_filtered, y_values_filtered, height, width, 8, 40, 0, 0)

if __name__ == "__main__":
    main()