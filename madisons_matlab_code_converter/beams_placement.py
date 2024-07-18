import numpy as np

# Example values for demonstration
c = np.array([1.3, 1.4, 1.5, 1.6])
y_nuc = np.array([2.6, 2.8, 3.0, 3.2])

# Adjust initial beam location to prevent initial overclosures
c_cyto = c * 1.01  # Slightly increase c to account for cytoskeleton

# Define x and y coordinates for beams (example data)
x_beam = np.array([0, 0, 0, 0])
y_beam = np.array([
    [1, 3.88, 5.91, 5.43],
    [2, 4.75, 7.03, 6.21],
    [3, 6.40, 8.53, 7.90],
    [4, 9.21, 9.74, 10.33],
    [5, 10.76, 10.42, 11.97],
    [6, 11.39, 11.54, 14.06],
    [7, 13.62, 12.97, 14.69],
    [8, 15.27, 14.11, 16.58],
    [9, 16.19, 15.71, 18.03],
    [-100, 18.08, 16.44, 20.02],
    [-100, -100, 19.63, -100]
])

# Iterate through each beam case
for i in range(len(x_beam)):
    # Find the last valid index
    k = np.where(y_beam[:, i] != -100)[0][-1] + 1

    # Find the center for phase shift
    y_center = (np.max(y_beam[:k, i]) - np.min(y_beam[:k, i])) / 2

    # Iterate through each y value up to k
    for j in range(k):
        # Calculate phase shift of beams: coordinate - center - initial offset
        y_0 = y_beam[j, i] - y_center - y_beam[0, i]

        # Scale beams so they do not fall off the nucleus
        scale = y_nuc[i] / (y_center * 2)
        y_scaled = scale * y_0 * 0.75  # Apply scaling factor

        # Calculate the squared value inside the square root
        val_inside_sqrt = (c_cyto[i]**2) * (1 - (y_scaled**2 / (0.8 * a[i])**2))

        # Check if the value inside the square root is non-negative
        if val_inside_sqrt >= 0:
            # Calculate the z values
            z_positive = np.sqrt(val_inside_sqrt)
            z_beam = z_positive

            # Display the results
            print(f'z is: {z_positive:.2f}')
        else:
            # If the value inside the square root is negative, x and y are not valid
            print('No valid z values for the given x, y, and r.')
