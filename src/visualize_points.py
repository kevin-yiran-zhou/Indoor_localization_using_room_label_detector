image_points = [[1236., 1235.],
 [1235.,  846.],
 [1751., 1237.],
 [1762.,  849.]]

import matplotlib.pyplot as plt

# Extract x and y coordinates from image_points
x_coords = [point[0] for point in image_points]
y_coords = [point[1] for point in image_points]

# Create a scatter plot
plt.scatter(x_coords, y_coords, color='blue')

# Add labels and title
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.title('Scatter Plot of Image Points')

# Show the plot
plt.show()