import numpy as np
from scipy.optimize import minimize

# Volumes of intact nuclei
nuclear_volumes = [894, 977, 805, 926]

# Ratios for semi-minor axes
b_ratio = 0.80
c_ratio = 0.65
cap_ratio = 0.01

# Initialize arrays to store results
a_values = np.zeros(len(nuclear_volumes))
b_values = np.zeros(len(nuclear_volumes))
c_values = np.zeros(len(nuclear_volumes))
h_values = np.zeros(len(nuclear_volumes))
nucleus_dimensions = np.zeros((len(nuclear_volumes), 3))  # [x_nuc, y_nuc, z_nuc]

for i in range(len(nuclear_volumes)):
    V_ellipsoid_no_cap = nuclear_volumes[i]

    # Define the volume equation for the ellipsoid with cap adjustment
    volume_equation = lambda a: ((4/3) * np.pi * b_ratio * c_ratio * a**3) * (1 - cap_ratio) - V_ellipsoid_no_cap

    # Objective function to minimize
    objective_function = lambda a: volume_equation(a)**2

    # Use minimize to find the semi-major axis 'a'
    result = minimize(objective_function, 1, bounds=[(0, None)])
    a_solution = result.x[0]

    # Calculate the semi-major axis 'a'
    a_values[i] = a_solution
    # Calculate the semi-minor axes 'b' and 'c'
    b_values[i] = b_ratio * a_solution
    c_values[i] = c_ratio * a_solution

    # Calculate the dimensions of the nucleus
    nucleus_dimensions[i] = [2 * a_solution, 2 * b_values[i], 2 * c_values[i]]

    # Calculate the volume of the ellipsoid
    V_ellipsoid = (4/3) * np.pi * a_solution * b_values[i] * c_values[i]
    V_cap = cap_ratio * V_ellipsoid

    # Define and solve the cap height equation
    cap_height_equation = lambda h: (np.pi * a_solution * b_values[i] * h**2 / (3 * c_values[i]**2)) * (3 * c_values[i] - h) - V_cap

    result_cap = minimize(lambda h: cap_height_equation(h)**2, 1, bounds=[(0, None)])
    h_solution = result_cap.x[0]

    # Extract the height of the cap
    h_values[i] = h_solution

# Display results
print('Semi-major axis (a):', a_values)
print('Semi-minor axis (b):', b_values)
print('Semi-minor axis (c):', c_values)
print('Height of the cap (h):', h_values)
print('Nucleus dimensions (x, y, z):', nucleus_dimensions)
