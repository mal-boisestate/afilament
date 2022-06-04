import pickle
import os
import glob
import cv2.cv2 as cv2
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import trimesh
from skimage import measure
import time
from colour import Color
import pickle
import pyvista as pv
import pymeshfix as mf
from afilament.objects.Node import run_node_creation
from afilament.objects.Parameters import ImgResolution


def get_nuc_mesh(cell):
    _, tmesh = save_mesh(cell.nucleus.nuc_3D, 'mesh_nuclei.stl')
    return tmesh

def save_mesh(image_3d, path):
    # idea was tacken here: https://forum.image.sc/t/3d-model-from-image-slices/33675/12
    p = 1
    z0 = np.zeros((p, image_3d.shape[1], image_3d.shape[2]), dtype=image_3d.dtype)
    z1 = np.zeros((image_3d.shape[0] + p * 2, p, image_3d.shape[2]), dtype=image_3d.dtype)
    image_3d = np.concatenate((z1, np.concatenate((z0, image_3d, z0), axis=0), z1), axis=1)

    verts, faces, normals, values = measure.marching_cubes(image_3d, 0, step_size=3, allow_degenerate=False)
    # for v in verts:
    #     v[2] *= - 1  #to flip nucleous
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)

    surf_mesh_smooth = trimesh.Trimesh.smoothed(surf_mesh)
    surf_mesh.export(path)
    #surf_mesh_smooth.export('mesh_nuclei_smooth.stl')
    print(f"The axis aligned box size of the nuclei: {surf_mesh.extents}")
    return image_3d, surf_mesh

def optimize_mesh(tmesh, scale):
    mesh = pv.wrap(tmesh)
    meshfix = mf.MeshFix(mesh)
    meshfix.repair(verbose=True)
    fixed_mesh = meshfix.mesh
    scaled_mesh = fixed_mesh.scale([1.0, 1.0, resolution.z / resolution.y], inplace=False)
    smoothed_mesh = scaled_mesh.smooth(n_iter=400)
    final_mesh = smoothed_mesh.subdivide(1, subfilter="loop")
    return final_mesh


def get_nodes(cell, resolution):
    fibers = cell.actin_total
    nodes, actin_fibers_w_nodes = run_node_creation(fibers.fibers_list, new_actin_len_th=2, is_plot_nodes=False)
    nodes_points = np.array([])
    for node in nodes:
        node_point = np.array([node.x, node.y, node.z])
        nodes_points = np.append(nodes_points, node_point)
    nodes_point_cloud = pv.PolyData(nodes_points)
    nodes_point_cloud_scaled = nodes_point_cloud.scale([1.0, 1.0, resolution.z / resolution.y], inplace=False)
    return actin_fibers_w_nodes, nodes, nodes_point_cloud_scaled

def draw_fibers_as_geopath(final_mesh, p, nodes, fibers, resolution):

    fibers_as_nodes_pairs ={}
    for i, node in enumerate(nodes):
        for id in node.actin_ids:
            if id in fibers_as_nodes_pairs:
                fibers_as_nodes_pairs[id].append(i)
                actin_xsection = fibers[id].get_stat(resolution)[1]
                fibers_as_nodes_pairs[id].append(actin_xsection) # add actin xsection as 3rd element of array
            else:
                nodes_in_fiber = [i]
                fibers_as_nodes_pairs[id] = nodes_in_fiber

def draw_nodes(p, nodes, resolution):
    nodes_points = np.array([])
    red = Color("red")
    colors = list(red.range_to(Color("green"), len(nodes)))
    for i, node in enumerate(nodes):
        color = colors[i].hex_l
        node_point = np.array([node.x, node.y, node.z * resolution.z / resolution.y])
        nodes_points = np.append(nodes_points, node_point)
        node_point = pv.PolyData(node_point)
        p.add_mesh(node_point, color=color, point_size=10.0, render_points_as_spheres=True)

def find_closest_nodes(mesh, nodes, resolution):
    closest_nodes_points = []
    for i, node in enumerate(nodes):
        node_point = np.array([node.x, node.y, node.z * resolution.z / resolution.y])
        closest_nodes_point = mesh.find_closest_point(node_point)
        closest_nodes_points.append(closest_nodes_point)
    return closest_nodes_points

def draw_closest_nodes(p, closest_nodes_indx, mesh):
    red = Color("red")
    colors = list(red.range_to(Color("green"), len(nodes)))
    for i, indx in enumerate(closest_nodes_indx):
        color = colors[i].hex_l
        p.add_mesh(mesh.points[indx], color=color, point_size=10.0, render_points_as_spheres=True)

def draw_fibers(mesh, p, cell):
    fibers = cell.actin_total.fibers_list
    red = Color("red")
    colors = list(red.range_to(Color("green"), len(fibers)))

    for color_indx, fiber in tqdm(enumerate(fibers)):
        actin_xsection = np.mean([cv2.contourArea(cnt) for cnt in fiber.cnts]) * resolution.y * resolution.z
        r = lambda: np.random.randint(0, 255)
        color = '#%02X%02X%02X' % (r(), r(), r())
        # color = colors[color_indx].hex_l
        for i in range(fiber.n):
            fiber_point = np.array([fiber.xs[i], fiber.ys[i], fiber.zs[i] * resolution.z / resolution.y])
            closest_fiber_point_indx = mesh.find_closest_point(fiber_point)
            p.add_mesh(mesh.points[closest_fiber_point_indx], color=color, point_size=actin_xsection*20, render_points_as_spheres=True, style='points')



if __name__ == '__main__':
    scale_x = 0.05882
    scale_y = 0.05882
    scale_z = 0.270
    resolution = ImgResolution(scale_x, scale_y, scale_z)
    file_path = r"D:\BioLab\Current_experiments\2022.06.02_very_first_cell_analysis_data\test_cells_bach.pickle"
    cells = pickle.load(open(file_path, "rb"))
    cell = cells[0]
    tmesh = get_nuc_mesh(cell)
    final_mesh = optimize_mesh(tmesh, resolution)
    fibers, nodes, nodes_point_cloud_scaled = get_nodes(cell, resolution)



    # closest_nodes_indx = find_closest_nodes(final_mesh, nodes, resolution)

    p = pv.Plotter()
    draw_fibers(final_mesh, p, cell)
    #
    # # draw_fibers_as_geopath(final_mesh, p, nodes, fibers, resolution)
    # # draw_nodes(p, nodes, resolution)
    # # draw_closest_nodes(p, closest_nodes_indx, final_mesh)
    p.add_mesh(final_mesh, color=True, show_edges=True)

    # p.add_mesh_clip_plane(final_mesh)

    p.show()