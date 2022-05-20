import os
import glob
import pickle
from collections import defaultdict
from tqdm import tqdm
import cv2.cv2 as cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import csv

from afilament.objects.SingleFiber import SingleFiber


class Node:

    def __init__(self, x, y, z, cnt, actin_id):
        self.x = x
        self.y = y
        self.z = z
        self.n = 1
        self.cnt = cnt
        self.actin_ids = [actin_id]
        self.geodesic_distance_from_center = None
        self.tg_xy = None
        self.ideal_x = None
        self.ideal_y = None
        self.ideal_z = None
        self.quadrant = None

    def get_stat(self, scale_y, scale_z):
        return [self.x, self.y, self.z,
                ",".join(map(str, self.actin_ids) ),
                cv2.contourArea(self.cnt) * scale_y * scale_z]

    def add_distance_from_center(self, geodesic_distance):
        self.geodesic_distance_from_center = geodesic_distance

    def add_xy_direction(self, tg_betta, quadrant):
        self.tg_xy = tg_betta
        self.quadrant = quadrant

    def add_ideal_coordinates(self, x, y, z):
        self.ideal_x = x
        self.ideal_y = y
        self.ideal_z = z


def plot_nodes(actin_fibers, nodes):
    ax = plt.axes(projection='3d')
    # ax = plt.axes()  #for 2D testing
    for fiber in actin_fibers:
        if len(np.unique(fiber.zs)) < 1:
            continue

        # Draw only center points
        xdata = fiber.xs
        ydata = fiber.ys
        zdata = fiber.zs
        color_x = 1.0 * np.random.randint(255) / 255
        color_y = 1.0 * np.random.randint(255) / 255
        color_z = 1.0 * np.random.randint(255) / 255
        if xdata:
            ax.scatter3D(xdata, ydata, zdata, c=[[color_x, color_y, color_z]] * len(xdata), cmap='Greens')

    xdata, ydata, zdata = [], [], []
    for node in nodes:
        xdata.append(node.x)
        ydata.append(node.y)
        zdata.append(node.z)
    color_x = 1.0
    color_y = 0
    color_z = 0
    if xdata:
        ax.scatter3D(xdata, ydata, zdata, c=[[color_x, color_y, color_z]] * len(xdata), s=50, cmap='Greens')

    plt.show()



def run_node_creation(actin_fibers, new_actin_len_th, is_plot_nodes):
    """
    new_actin_len_th - do not breake actin if one of the part is too small
    """

    nodes = []
    for actin_id, actin in enumerate(actin_fibers):
        nodes.append(Node(actin.xs[0], actin.ys[0], actin.zs[0], actin.cnts[0], actin_id))  # left side
        nodes.append(Node(actin.xs[-1], actin.ys[-1], actin.zs[-1], actin.cnts[-1], actin_id))  # right side
        actin.left_node_id = len(nodes) - 2
        actin.right_node_id = len(nodes) - 1

    for left_node in nodes[::2]:
        actin_id_to_break = None

        actins_to_check = [[actin_id, actin] for actin_id, actin in enumerate(actin_fibers)
                           if left_node.x - 1 in actin.xs]
        cnts_to_check = [actin.cnts[actin.xs.index(left_node.x - 1)] for _, actin in actins_to_check]

        node_mask = np.zeros((1300, 1300), dtype=np.uint8)
        cv2.drawContours(node_mask, [left_node.cnt], -1, 255, -1)
        for cnt_id, cnt in enumerate(cnts_to_check):
            cnt_mask = np.zeros((1300, 1300), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

            if np.any(cv2.bitwise_and(node_mask, cnt_mask)):
                actin_id_to_break = actins_to_check[cnt_id][0]

        if actin_id_to_break is not None:
            actin_to_break = actin_fibers[actin_id_to_break]
            break_index = actin_to_break.xs.index(left_node.x - 1)

            # do not break actin if one of the parts is too small
            if break_index < new_actin_len_th or len(actin_to_break.xs) - new_actin_len_th < break_index:
                continue

            new_right_actin = SingleFiber(-1, -1, -1, -1, -1)
            new_right_actin.xs = actin_to_break.xs[break_index + 1:]
            new_right_actin.ys = actin_to_break.ys[break_index + 1:]
            new_right_actin.zs = actin_to_break.zs[break_index + 1:]
            new_right_actin.last_layer = actin_to_break.last_layer[break_index + 1:]
            new_right_actin.cnts = actin_to_break.cnts[break_index + 1:]
            new_right_actin.n = len(new_right_actin.xs)
            new_right_actin.left_node_id = nodes.index(left_node)
            new_right_actin.right_node_id = actin_to_break.right_node_id

            actin_to_break.xs = actin_to_break.xs[:break_index + 1]
            actin_to_break.ys = actin_to_break.ys[:break_index + 1]
            actin_to_break.zs = actin_to_break.zs[:break_index + 1]
            actin_to_break.last_layer = actin_to_break.last_layer[:break_index + 1]
            actin_to_break.cnts = actin_to_break.cnts[:break_index + 1]
            actin_to_break.n = len(actin_to_break.xs)

            actin_fibers.append(new_right_actin)

            # update old actin
            actin_to_break.right_node_id = nodes.index(left_node)

            # update node
            left_node.actin_ids.append(actin_id_to_break)  # adding old actin which attached from left
            left_node.actin_ids.append(len(actin_fibers) - 1)  # adding new actin which attached from right

            # update old right node
            old_right_node = nodes[new_right_actin.right_node_id]
            old_right_node.actin_ids.pop(old_right_node.actin_ids.index(actin_id_to_break))
            old_right_node.actin_ids.append(len(actin_fibers) - 1)

    # -------------------------------------------------------- #
    for right_node in nodes[1::2]:
        actin_id_to_break = None

        actins_to_check = [[actin_id, actin] for actin_id, actin in enumerate(actin_fibers)
                           if right_node.x + 1 in actin.xs]
        cnts_to_check = [actin.cnts[actin.xs.index(right_node.x + 1)] for _, actin in actins_to_check]

        node_mask = np.zeros((1300, 1300), dtype=np.uint8)
        cv2.drawContours(node_mask, [right_node.cnt], -1, 255, -1)
        for cnt_id, cnt in enumerate(cnts_to_check):
            cnt_mask = np.zeros((1300, 1300), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

            if np.any(cv2.bitwise_and(node_mask, cnt_mask)):
                actin_id_to_break = actins_to_check[cnt_id][0]

        if actin_id_to_break is not None:
            actin_to_break = actin_fibers[actin_id_to_break]
            break_index = actin_to_break.xs.index(right_node.x + 1)

            # do not break actin if one of the parts is too small
            if break_index < new_actin_len_th or len(actin_to_break.xs) - new_actin_len_th < break_index:
                continue

            new_right_actin = SingleFiber(-1, -1, -1, -1, -1)
            new_right_actin.xs = actin_to_break.xs[break_index + 1:]
            new_right_actin.ys = actin_to_break.ys[break_index + 1:]
            new_right_actin.zs = actin_to_break.zs[break_index + 1:]
            new_right_actin.last_layer = actin_to_break.last_layer[break_index + 1:]
            new_right_actin.cnts = actin_to_break.cnts[break_index + 1:]
            new_right_actin.n = len(new_right_actin.xs)
            new_right_actin.left_node_id = nodes.index(right_node)
            new_right_actin.right_node_id = actin_to_break.right_node_id

            actin_to_break.xs = actin_to_break.xs[:break_index + 1]
            actin_to_break.ys = actin_to_break.ys[:break_index + 1]
            actin_to_break.zs = actin_to_break.zs[:break_index + 1]
            actin_to_break.last_layer = actin_to_break.last_layer[:break_index + 1]
            actin_to_break.cnts = actin_to_break.cnts[:break_index + 1]
            actin_to_break.n = len(actin_to_break.xs)

            actin_fibers.append(new_right_actin)

            # update old actin
            actin_to_break.right_node_id = nodes.index(right_node)

            # update node
            right_node.actin_ids.append(actin_id_to_break)  # adding old actin which attached from left
            right_node.actin_ids.append(len(actin_fibers) - 1)  # adding new actin which attached from right

            # update old right node
            old_right_node = nodes[new_right_actin.right_node_id]
            old_right_node.actin_ids.pop(old_right_node.actin_ids.index(actin_id_to_break))
            old_right_node.actin_ids.append(len(actin_fibers) - 1)
    if is_plot_nodes:
        plot_nodes(actin_fibers, nodes)
    return nodes, actin_fibers