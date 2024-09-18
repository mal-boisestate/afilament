from __future__ import annotations

import pyvista as pv
import numpy as np
from scipy.optimize import fsolve
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


def remesh_with_uniform_triangles(mesh, num_clusters=8000):
    # Ensure the mesh is fully triangulated before using pymeshfix
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()

    # Repair the mesh using pymeshfix
    points = np.array(mesh.points)
    faces = np.array(mesh.faces).reshape(-1, 4)[:, 1:]  # Extract triangular face indices
    meshfix = pymeshfix.MeshFix(points, faces)
    meshfix.repair()

    # Convert the repaired mesh back to pyvista format
    repaired_mesh = pv.PolyData(meshfix.v, np.hstack([[3] + list(face) for face in meshfix.f]))

    # Remesh the entire geometry uniformly using pyacvd
    cluster = pyacvd.Clustering(repaired_mesh)
    cluster.cluster(num_clusters)  # Set the number of clusters to control triangle size
    remeshed = cluster.create_mesh()

    return remeshed


def create_nucleus_mesh(nucleus_volume, num_clusters=8000):
    # Calculate parameters
    a, b, c, h = calculate_nucleus_parameters(nucleus_volume)

    # Start with a sphere and then scale the vertices to form an ellipsoid
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.0)

    # Scale the vertices by a, b, and c to form an ellipsoid
    vertices = sphere.vertices
    vertices[:, 0] *= a  # scale x by a
    vertices[:, 1] *= b  # scale y by b
    vertices[:, 2] *= c  # scale z by c

    # Get the faces of the mesh
    faces = sphere.faces

    # Convert to pyvista mesh
    ellipsoid_mesh = pv.PolyData(vertices, np.hstack([[3] + list(face) for face in faces]))

    # Ensure the ellipsoid mesh is fully triangulated
    ellipsoid_mesh = ellipsoid_mesh.triangulate()

    # Remesh the ellipsoid uniformly before clipping
    remeshed_ellipsoid = remesh_with_uniform_triangles(ellipsoid_mesh, num_clusters)

    # Clip the ellipsoid with the plane at z = -(c - h)
    z_coordinate = -(c - h)
    clipped_ellipsoid = remeshed_ellipsoid.clip('z', origin=(0, 0, z_coordinate), invert=False)

    # Cap the bottom of the ellipsoid with a plane
    plane = pv.Plane(center=(0, 0, z_coordinate), direction=(0, 0, 1), i_size=2 * a, j_size=2 * b)
    plane = plane.triangulate()

    # Combine the clipped ellipsoid with the plane
    combined = clipped_ellipsoid.boolean_union(plane)

    return combined


# Example usage:
nucleus = create_nucleus_mesh(977, num_clusters=8000)

# Plot the nucleus with uniform triangles
plotter = pv.Plotter()
plotter.add_mesh(nucleus, color='lightblue', show_edges=True)
plotter.show()
