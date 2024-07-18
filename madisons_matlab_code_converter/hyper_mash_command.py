# Importing necessary libraries
import numpy as np

# Assuming h and c are already defined arrays
h = np.array([0.5, 1.0, 1.5, 2.0])  # Example heights of the caps
c = np.array([2.0, 2.5, 3.0, 3.5])  # Example semi-minor axes

# Initialize an array to store h_cap values
h_cap = np.zeros(len(h))

# Generating command line for hypermesh
for i in range(len(h)):
    # Calculate the height of the cap
    h_cap[i] = -(c[i] - h[i])

    # String used before coordinates within hypermesh
    plane_begin_string = '*surfacemode 4\n*createplane 1 '
    plane_end_string = ' 0 0 ' + str(h_cap[i]) + '\n*surfaceplane 1 50'

    # Calling function
    hypermesh_command_line = (plane_begin_string + ' ' + str(0) + ' ' +
                              str(0) + ' ' + str(h_cap[i]) + ' ' + plane_end_string)

    # Printing function
    print(f'\nHEIGHT OF NUCLEI {i+1}')
    print(hypermesh_command_line)
