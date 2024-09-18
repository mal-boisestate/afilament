from __future__ import annotations

import pyvista as pv
import numpy as np
from scipy.optimize import fsolve
import vtk

def calculate_nucleus_parameters(nucleus_volume, b_ratio=0.80, c_ratio=0.65, cap_ratio=0.01):
    # Define the volume equation for the full ellipsoid
    def volume_equation(a):
        return (4 / 3) * np.pi * a * (b_ratio * a) * (c_ratio * a) - nucleus_volume

    # Use fsolve to find the semi-major axis 'a' for the full ellipsoid
    a_solution = fsolve(volume_equation, 1)[0]

    # Calculate the semi-major axis 'a'
    a = a_solution
    # Calculate the semi-minor axes 'b' and 'c'
    b = b_ratio * a
    c = c_ratio * a

    # Calculate the volume of the ellipsoid
    V_ellipsoid = (4 / 3) * np.pi * a * b * c
    # Cap volume is 1% of the ellipsoid volume
    V_cap = cap_ratio * V_ellipsoid

    # Define and solve the cap height equation
    def cap_height_equation(h):
        return (np.pi * a * b * h ** 2 / (3 * c ** 2)) * (3 * c - h) - V_cap

    h_solution = fsolve(cap_height_equation, 0.1)[0]

    # Extract the height of the cap
    h = h_solution

    return a, b, c, h

def create_nucleus_mesh(nucleus_volume):
    # Calculate parameters
    a, b, c, h = calculate_nucleus_parameters(nucleus_volume)

    # Create the ellipsoid centered at the origin
    ellipsoid = pv.ParametricEllipsoid(a, b, c, center=(0, 0, 0)).triangulate()

    # Calculate the z-coordinate to cut the ellipsoid
    z_coordinate = -(c - h)

    # Clip the ellipsoid with the plane, keeping the part below the plane
    ellipsoid_clipped = ellipsoid.clip('z', origin=(0, 0, z_coordinate), invert=False)

    # Create a plane at the clipping z-coordinate
    plane = pv.Plane(center=(0, 0, z_coordinate), direction=(0, 0, 1), i_size=2 * a, j_size=2 * b)

    # Triangulate the plane
    plane = plane.triangulate()

    # Combine the clipped ellipsoid with the plane
    combined = plane.boolean_difference(ellipsoid_clipped)

    # Triangulate the combined mesh to ensure it is a proper solid
    nucleus = combined.triangulate()

    # Convert the surface mesh to a volume mesh using vtkDelaunay3D
    delny = vtk.vtkDelaunay3D()
    delny.SetInputData(nucleus)
    delny.Update()

    # Extract the unstructured grid
    ugrid = pv.wrap(delny.GetOutput())

    return ugrid

def create_beam_mesh_list(nucleus_volume, y_list, beam_increase_ratio=1.01):
    beam_mesh_list = []
    node_points = []
    line_points = []
    a, b, c, h = calculate_nucleus_parameters(nucleus_volume * beam_increase_ratio)
    beam_width = 0.3
    z_min = c / 3 * 2  # This is preliminary placement that should be justified
    z_max = c + 2  # To make sure that the bounding box is higher than the nucleus

    # Create the ellipsoid centered at the origin
    ellipsoid = pv.ParametricEllipsoid(a, b, c, center=(0, 0, 0))

    for y in y_list:
        beam_bounds = [-100, 100, y - beam_width / 2, y + beam_width / 2, z_min, z_max]
        beam = ellipsoid.clip_box(bounds=beam_bounds, invert=False)
        # Convert beam to PolyData for decimation
        beam_polydata = beam.extract_surface()
        # Calculate target number of elements
        target_number_of_elements = 50  # Define an appropriate target
        reduction_factor = 1 - (target_number_of_elements / beam_polydata.n_cells)
        # Remesh the beam for optimal mesh quality
        remeshed_beam = beam_polydata.decimate(target_reduction=reduction_factor)
        beam_mesh_list.append(remeshed_beam)

        # Find the y-coordinates of the leftmost and rightmost nodes
        y_min, y_max = remeshed_beam.bounds[2], remeshed_beam.bounds[3]
        # Create the nodes
        left_node = [-20, y_min, -(c - h)]
        right_node = [20, y_max, -(c - h)]
        node_points.extend([left_node, right_node])

        # Find the actual leftmost and rightmost nodes of the beam
        leftmost_node = remeshed_beam.points[np.argmin(remeshed_beam.points[:, 0])]
        rightmost_node = remeshed_beam.points[np.argmax(remeshed_beam.points[:, 0])]

        # Add lines connecting the leftmost and rightmost nodes to the created nodes
        line_points.append([leftmost_node, left_node])
        line_points.append([rightmost_node, right_node])

    return beam_mesh_list, node_points, line_points

nucleus = create_nucleus_mesh(977)
y_positions = [0, -2, 2, 3.5, -3.5, -1, 1 ]
beam_mesh_list, node_points, line_points = create_beam_mesh_list(977, y_positions)

plotter = pv.Plotter()

plotter.add_mesh(nucleus, color='lightblue', show_edges=True)
for beam in beam_mesh_list:
    plotter.add_mesh(beam, color='red', opacity=0.5)

# Add node points as small yellow spheres
nodes = pv.PolyData(np.array(node_points))
plotter.add_mesh(nodes, color='yellow', point_size=10, render_points_as_spheres=True)

# Add lines connecting the nodes
for line in line_points:
    line_mesh = pv.Line(line[0], line[1])
    plotter.add_mesh(line_mesh, color='yellow', line_width=2)

plotter.render()



# Display the ellipsoid and the clipped part
plotter.show()
