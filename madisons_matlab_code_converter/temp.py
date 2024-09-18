from __future__ import annotations

import pyvista as pv
import numpy as np
from scipy.optimize import fsolve
import vtk
import trimesh
import pyacvd
import pymeshfix

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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

def create_nucleus_mesh(nucleus_volume, subdivisions=4):
    # Calculate parameters
    a, b, c, h = calculate_nucleus_parameters(nucleus_volume)

    # Start with a sphere and then scale the vertices to form an ellipsoid
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)

    # Scale the vertices by a, b, and c to form an ellipsoid
    vertices = sphere.vertices
    vertices[:, 0] *= a  # scale x by a
    vertices[:, 1] *= b  # scale y by b
    vertices[:, 2] *= c  # scale z by c
    # Get the faces of the mesh (unchanged)
    faces = sphere.faces

    # Pyvista expects faces in a flat array where the first value is the number of vertices per face
    faces_pv = np.hstack([[3] + list(face) for face in faces])

    # Create the mesh using pyvista
    ellipsoid = pv.PolyData(vertices, faces_pv)

    # Calculate the z-coordinate to cut the ellipsoid
    z_coordinate = -(c - h)

    # Clip the ellipsoid with the plane, keeping the part below the plane
    ellipsoid_clipped = ellipsoid.clip('z', origin=(0, 0, z_coordinate), invert=False)
    nucleus = ellipsoid_clipped

    # Create a plane at the clipping z-coordinate
    plane = pv.Plane(center=(0, 0, z_coordinate), direction=(0, 0, 1), i_size=2 * a, j_size=2 * b)

    # Triangulate the plane
    plane = plane.triangulate()

    # Combine the clipped ellipsoid with the plane
    combined = ellipsoid_clipped.boolean_difference(plane)

    # Triangulate the combined mesh to ensure it is a proper solid
    nucleus_pv = combined.triangulate()

    # Use PyMeshFix to remesh the clipped ellipsoid to have uniform triangle sizes
    # This will fix any issues in the mesh and remesh with uniform elements
    points = np.array(nucleus_pv.points)
    faces = np.array(nucleus_pv.faces).reshape(-1, 4)[:, 1:]  # Extract face indices (ignore the leading "3")

    # Create the PyMeshFix object from points and faces
    meshfix = pymeshfix.MeshFix(points, faces)
    meshfix.repair()  # Repair and remesh the mesh

    # Access repaired points and faces from PyMeshFix
    repaired_points = meshfix.v  # Repaired vertices
    repaired_faces = meshfix.f  # Repaired faces

    # Convert back to PyVista format for visualization
    repaired_mesh = pv.PolyData(repaired_points, np.hstack([[3] + list(face) for face in repaired_faces]))

    # Step 2: Remesh using pyacvd for uniform triangle sizes
    cluster = pyacvd.Clustering(repaired_mesh)
    cluster.cluster(8000)  # Adjust this number to control triangle density
    remeshed_nucleus = cluster.create_mesh()

    # # Convert the surface mesh to a volume mesh using vtkDelaunay3D
    # delny = vtk.vtkDelaunay3D()
    # delny.SetInputData(remeshed_nucleus)
    # delny.Update()
    #
    # # Extract the unstructured grid
    # ugrid = pv.wrap(delny.GetOutput())

    return remeshed_nucleus


def create_beam_mesh_list(nucleus_volume, y_list, beam_increase_ratio=1.02, subdivisions=3):
    beam_mesh_list = []
    node_points = []
    line_points = []
    a, b, c, h = calculate_nucleus_parameters(nucleus_volume * beam_increase_ratio)
    beam_width = 0.3
    z_min = c / 3 * 2  # Preliminary placement
    z_max = c + 2  # To ensure the bounding box is higher than the nucleus

    # Start with a sphere and scale vertices to form an ellipsoid
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    vertices = sphere.vertices
    vertices[:, 0] *= a  # scale x by a
    vertices[:, 1] *= b  # scale y by b
    vertices[:, 2] *= c  # scale z by c
    faces = sphere.faces

    # Pyvista expects faces in a flat array where the first value is the number of vertices per face
    faces_pv = np.hstack([[3] + list(face) for face in faces])
    ellipsoid = pv.PolyData(vertices, faces_pv)

    for y in y_list:
        # Define the beam bounds
        beam_bounds = [-100, 100, y - beam_width / 2, y + beam_width / 2, z_min, z_max]

        # Clip the ellipsoid to get the beam
        beam = ellipsoid.clip_box(bounds=beam_bounds, invert=False)

        # Extract the surface of the beam for further processing
        beam_polydata = beam.extract_surface()

        # Calculate target number of elements
        target_number_of_elements = 50  # Define an appropriate target
        reduction_factor = max(0.0, min(1.0, 1 - (target_number_of_elements / beam_polydata.n_cells)))

        # Decimate the mesh with the calculated reduction factor
        if beam_polydata.n_cells > target_number_of_elements:
            remeshed_beam = beam_polydata.decimate(target_reduction=reduction_factor)
        else:
            remeshed_beam = beam_polydata

        beam_mesh_list.append(remeshed_beam)

        # Get all points from the beam
        points = remeshed_beam.points

        # Ensure unique selection of points
        # Find the leftmost bottom point (minimum x, minimum y)
        left_most_bottom = points[np.argmin(points[:, 0] + points[:, 1])]

        # Find the leftmost top point (minimum x, maximum y) but different from left_most_bottom
        left_most_top_candidates = points[points[:, 1] > left_most_bottom[1]]
        if len(left_most_top_candidates) > 0:
            left_most_top = left_most_top_candidates[np.argmin(left_most_top_candidates[:, 0])]
        else:
            left_most_top = left_most_bottom  # Fallback to bottom if no candidates

        # Find the rightmost bottom point (maximum x, minimum y)
        right_most_bottom = points[np.argmax(points[:, 0] - points[:, 1])]

        # Find the rightmost top point (maximum x, maximum y) but different from right_most_bottom
        right_most_top_candidates = points[points[:, 1] > right_most_bottom[1]]
        if len(right_most_top_candidates) > 0:
            right_most_top = right_most_top_candidates[np.argmax(right_most_top_candidates[:, 0])]
        else:
            right_most_top = right_most_bottom  # Fallback to bottom if no candidates

        # Create the nodes based on the calculated points
        left_node = [-20, (left_most_bottom[1] + left_most_top[1]) / 2, -(c - h)]
        right_node = [20, (right_most_bottom[1] + right_most_top[1]) / 2, -(c - h)]
        node_points.extend([left_node, right_node])

        # Add lines connecting the leftmost and rightmost nodes to the created nodes
        if not np.array_equal(left_most_bottom, left_most_top):
            line_points.append([left_most_bottom, left_node])
            line_points.append([left_most_top, left_node])
        else:
            # Handle degenerate case where top and bottom are the same
            line_points.append([left_most_bottom, left_node])

        if not np.array_equal(right_most_bottom, right_most_top):
            line_points.append([right_most_bottom, right_node])
            line_points.append([right_most_top, right_node])
        else:
            # Handle degenerate case where top and bottom are the same
            line_points.append([right_most_bottom, right_node])

    return beam_mesh_list, node_points, line_points


nucleus = create_nucleus_mesh(977)
y_positions = [0, -2, 2, 3.5, -3.5, -1, 1 ]
beam_mesh_list, node_points, line_points = create_beam_mesh_list(977, y_positions)

plotter = pv.Plotter()

plotter.add_mesh(nucleus, color='lightblue', show_edges=False)
for beam in beam_mesh_list:
    plotter.add_mesh(beam, color='red', opacity=0.5, show_edges=True)

# Add node points as small yellow spheres
nodes = pv.PolyData(np.array(node_points))
plotter.add_mesh(nodes, color='yellow', point_size=10, render_points_as_spheres=True)

# Add lines connecting the nodes
for line in line_points:
    line_mesh = pv.Line(line[0], line[1])
    plotter.add_mesh(line_mesh, color='yellow', line_width=2)

plotter.render()

plotter.show()
