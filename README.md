# Image Graph Analysis with Python

## Overview

This repository contains a Python script designed for the analysis and processing of images containing graphs. The script employs a systematic approach involving various image processing techniques to identify, filter, and scale relevant information within the graph. The process ensures accurate extraction of numerical data from visual graph representations.

## Approach

1. **Preprocessing:**
   - Grayscale conversion
   - Gaussian blur
   - Adaptive thresholding

2. **Dynamic Cropping:**
   - Focuses on isolating the line graph by identifying the bounding box of the largest contour.

3. **Line Detection and Filtering:**
   - Canny edge detection
   - Probabilistic Hough Transform to find lines
   - Filtering lines based on slope to exclude horizontal and vertical lines

4. **Data Point Extraction:**
   - Extracting data points from the filtered lines

5. **Data Point Filtering:**
   - Filtering data points based on the angle between vectors

6. **Visualization:**
   - Displaying visualizations at different stages, including the original image with detected lines

7. **Conversion to Original Coordinates:**
   - Calculating scaling factors and adjusting data points to convert to original coordinates

8. **Final Graph Visualization:**
   - Displaying the final graph with modified and scaled data points

## Input
![Alt Text](/sample%20images/line_graph.png)


## Output
![Alt Text](/output%20images/cropped_image.jpeg)

![Alt Text](/output%20images/detected_lines.jpeg)

![Alt Text](/output%20images/filtered_lines.jpeg)

![Alt Text](/output%20images/filtered_datapoints.jpeg)

![Alt Text](/output%20images/final_graph.jpeg)



## Extracted Values
![Alt Text](/output%20images/data_values.jpeg)


## License

This project is licensed under the [MIT License](LICENSE).
