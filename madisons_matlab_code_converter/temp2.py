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

    repaired_mesh=repaired_mesh.subdivide_adaptive(max_edge_len=1, max_n_passes=3)

    # Step 2: Remesh using pyacvd for uniform triangle sizes
    # mesh is not dense enough for uniform remeshing

    cluster = pyacvd.Clustering(repaired_mesh)
    cluster.subdivide(3)
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

def create_beam_mesh_list(nucleus_volume, y_list, beam_increase_ratio=1.02, subdivisions=4):
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

        def triangulate_with_fixed_boundary(polydata, target_edge_length):
            # Step 1: Extract boundary points
            boundary_edges = vtk.vtkFeatureEdges()
            boundary_edges.SetInputData(polydata)
            boundary_edges.BoundaryEdgesOn()
            boundary_edges.FeatureEdgesOff()
            boundary_edges.NonManifoldEdgesOff()
            boundary_edges.ManifoldEdgesOff()
            boundary_edges.Update()

            boundary_polydata = boundary_edges.GetOutput()

            # Step 2: Use vtkDelaunay2D to create a triangulation constrained to the boundary
            delaunay = vtk.vtkDelaunay2D()
            delaunay.SetInputData(boundary_polydata)
            delaunay.Update()

            # Step 3: Use vtkTriangleFilter to ensure all faces are triangles
            triangle_filter = vtk.vtkTriangleFilter()
            triangle_filter.SetInputConnection(delaunay.GetOutputPort())
            triangle_filter.Update()

            # Step 4: Apply vtkDecimatePro to control the number of triangles and approximate edge length
            decimate = vtk.vtkDecimatePro()
            decimate.SetInputConnection(triangle_filter.GetOutputPort())
            decimate.SetTargetReduction(
                1.0 - (target_edge_length / max(target_edge_length, 1.0)))  # Approximate control
            decimate.PreserveTopologyOn()  # Ensure the topology is preserved
            decimate.Update()

            # Final remeshed output
            remeshed_polydata = decimate.GetOutput()

            return remeshed_polydata

        # Convert VTK PolyData to PyVista format
        def vtk_to_pyvista(vtk_polydata):
            return pv.wrap(vtk_polydata)

        # Assuming beam_polydata is your original boundary-constrained geometry
        remeshed_beam_vtk = triangulate_with_fixed_boundary(beam_polydata, target_edge_length=0.4)

        # Convert to PyVista format
        remeshed_beam = vtk_to_pyvista(remeshed_beam_vtk)

        # The new meshed beam is now in PyVista format
        beam_mesh_list.append(remeshed_beam)

        # Get all points from the beam
        points = remeshed_beam.points

        # Separate points into four quadrants
        left_candidates = points[points[:, 0] <= 0]
        right_candidates = points[points[:, 0] > 0]
        bottom_candidates = points[points[:, 1] <= np.median(points[:, 1])]
        top_candidates = points[points[:, 1] > np.median(points[:, 1])]

        # Handle left-bottom corner: min x and min y
        left_bottom_candidates = left_candidates[np.isin(left_candidates, bottom_candidates).all(axis=1)]
        if len(left_bottom_candidates) > 0:
            left_most_bottom = left_bottom_candidates[np.argmin(left_bottom_candidates[:, 0] + left_bottom_candidates[:, 1])]
        else:
            left_most_bottom = left_candidates[np.argmin(left_candidates[:, 0] + left_candidates[:, 1])]

        # Handle left-top corner: min x and max y
        left_top_candidates = left_candidates[np.isin(left_candidates, top_candidates).all(axis=1)]
        if len(left_top_candidates) > 0:
            left_most_top = left_top_candidates[np.argmin(left_top_candidates[:, 0] - left_top_candidates[:, 1])]
        else:
            left_most_top = left_candidates[np.argmax(left_candidates[:, 1])]

        # Handle right-bottom corner: max x and min y
        right_bottom_candidates = right_candidates[np.isin(right_candidates, bottom_candidates).all(axis=1)]
        if len(right_bottom_candidates) > 0:
            right_most_bottom = right_bottom_candidates[np.argmax(right_bottom_candidates[:, 0] - right_bottom_candidates[:, 1])]
        else:
            right_most_bottom = right_candidates[np.argmin(right_candidates[:, 0] + right_candidates[:, 1])]

        # Handle right-top corner: max x and max y
        right_top_candidates = right_candidates[np.isin(right_candidates, top_candidates).all(axis=1)]
        if len(right_top_candidates) > 0:
            right_most_top = right_top_candidates[np.argmax(right_top_candidates[:, 0] + right_top_candidates[:, 1])]
        else:
            right_most_top = right_candidates[np.argmax(right_candidates[:, 1])]

        # Create the nodes based on the calculated points
        left_node = [-20, (left_most_bottom[1] + left_most_top[1]) / 2, -(c - h)]
        right_node = [20, (right_most_bottom[1] + right_most_top[1]) / 2, -(c - h)]
        node_points.extend([left_node, right_node])

        # Add lines connecting the leftmost and rightmost nodes to the created nodes
        line_points.append([left_most_bottom, left_node])
        line_points.append([left_most_top, left_node])
        line_points.append([right_most_bottom, right_node])
        line_points.append([right_most_top, right_node])

    return beam_mesh_list, node_points, line_points




nucleus = create_nucleus_mesh(977)
y_positions = [0, -2, 2, 3.5, -3.5, -1, 1 ]
beam_mesh_list, node_points, line_points = create_beam_mesh_list(977, y_positions)

plotter = pv.Plotter()

plotter.add_mesh(nucleus, color='lightblue', show_edges=True)
for beam in beam_mesh_list:
    plotter.add_mesh(beam, color='red', opacity=0.5, show_edges=False)

# Add node points as small yellow spheres
nodes = pv.PolyData(np.array(node_points))
plotter.add_mesh(nodes, color='yellow', point_size=10, render_points_as_spheres=True)

# Add lines connecting the nodes
for line in line_points:
    line_mesh = pv.Line(line[0], line[1])
    plotter.add_mesh(line_mesh, color='yellow', line_width=2)

plotter.render()

plotter.show()

