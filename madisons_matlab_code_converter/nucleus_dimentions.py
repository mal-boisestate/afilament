import numpy as np
from scipy.optimize import fsolve

def calculate_nucleus_parameters(nucleus_volume, b_ratio=0.80, c_ratio=0.65, cap_ratio=0.01):
    # Define the volume equation for the full ellipsoid
    def volume_equation(a):
        return (4/3) * np.pi * a * (b_ratio * a) * (c_ratio * a) - nucleus_volume

    # Use fsolve to find the semi-major axis 'a' for the full ellipsoid
    a_solution = fsolve(volume_equation, 1)[0]

    # Calculate the semi-major axis 'a'
    a = a_solution
    # Calculate the semi-minor axes 'b' and 'c'
    b = b_ratio * a
    c = c_ratio * a

    # Calculate the dimensions of the nucleus
    x_nuc = 2 * a
    y_nuc = 2 * b
    z_nuc = 2 * c
    nucleus_dimensions = [x_nuc, y_nuc, z_nuc]

    # Calculate the volume of the ellipsoid
    V_ellipsoid = (4/3) * np.pi * a * b * c
    # Cap volume is 1% of the ellipsoid volume
    V_cap = cap_ratio * V_ellipsoid

    # Define and solve the cap height equation
    def cap_height_equation(h):
        return (np.pi * a * b * h**2 / (3 * c**2)) * (3 * c - h) - V_cap

    h_solution = fsolve(cap_height_equation, 0.1)[0]

    # Extract the height of the cap
    h = h_solution

    return a, b, c, h, nucleus_dimensions

# Example usage
nucleus_volume = 894

# Calculate nucleus parameters
a, b, c, h, nucleus_dimensions = calculate_nucleus_parameters(nucleus_volume)

# Display results
print('Semi-major axis (a):', a)
print('Semi-minor axis (b):', b)
print('Semi-minor axis (c):', c)
print('Height of the cap (h):', h)
print('Nucleus dimensions (x, y, z):', nucleus_dimensions)
