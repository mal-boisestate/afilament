import csv
import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics
from collections import defaultdict
from afilament.objects import Utils
from unet.predict import run_predict_unet
from afilament.objects import Contour
from afilament.objects.SingleFiber import SingleFiber
from afilament.objects.Node import Node, add_edge_nodes
from afilament.objects.MergedFiber import MergedFiber


class ActinContour(object):
    def __init__(self, x, y, z, cnt):
        self.x = x
        self.y = y
        self.z = z
        self.cnt = cnt
        self.xsection = 0
        self.parent = None


class Fibers(object):
    # This object keeps all information about the cell that I need for analysis
    def __init__(self, part):
        self.part = part
        self.total_volume = None
        self.total_length = None
        self.total_num = None
        self.fibers_list = None
        self.slope_variance = None
        self.is_merged = False

    def get_actin_cnt_objs(self, xsection, x):
        cnts = cv2.findContours(xsection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        actin_cnt_objs = []
        for cnt in cnts:
            z, y = Contour.get_cnt_center(cnt)
            actin_cnt_objs.append(ActinContour(x, y, z, cnt))

        return actin_cnt_objs

    def _get_actin_fibers(self, img_3d, fiber_min_layers_theshold):
        """
        Creates initial actin fibers based on the biggest intersection of actin contour on successive layers.

        Finds all contours of actin fibers on each yz layer and then add each of the contours to the existing actin fiber
        object if specified contour overlaps some contour from the previous layer. If the contour does not overlap any other
        contour from the previous layer, a new actin fiber object for this contour is created. If the contour overlaps more
        than one contours from the previous layer, the contour will be added to a fiber whose contour has the biggest
        overlap area. If two contours on the current layer overlap the same contour on the previous layer the contour with a
        bigger overlapping area will be added to the existed actin fiber object while a new actin fiber object would be
        created for the second contour.
        ---
            Parameters:
            - img_3d (np.array): A three-dimensional numpy array which represents a 3D image mask of actin fibers.
        ---
            Returns:
            - actin_fibers (List[ActinFaber]): List of ActinFiber objects
        """
        actin_fibers = []

        for x_slice in range(img_3d.shape[0]):
            print("Processing {} slice out of {} slices".format(x_slice, img_3d.shape[0]))

            xsection = img_3d[x_slice, :, :]

            actin_cnt_objs = self.get_actin_cnt_objs(xsection, x_slice)

            if x_slice == 0 or not actin_fibers:
                actin_fibers.extend([SingleFiber(actin_contour_obj.x,
                                                 actin_contour_obj.y,
                                                 actin_contour_obj.z,
                                                 x_slice,
                                                 actin_contour_obj.cnt)
                                     for actin_contour_obj in actin_cnt_objs])
            else:
                actin_fibers_from_previous_layer = [fiber for fiber in actin_fibers if
                                                    fiber.last_layer[-1] == x_slice - 1]
                print("Number of actins from previous layer: {}".format(len(actin_fibers_from_previous_layer)))

                # find parents for all contours on new layer
                # is_increased_intersection is increased by 1 pixel to check if it helps to reconstruct fibers more precise
                for new_layer_cnt_obj in actin_cnt_objs:
                    for i, actin_fiber in enumerate(actin_fibers_from_previous_layer):
                        new_layer_cnt_mask = np.zeros_like(xsection)
                        cv2.drawContours(new_layer_cnt_mask, [new_layer_cnt_obj.cnt], -1, 255, -1)

                        actin_cnt_mask = np.zeros_like(xsection)
                        cv2.drawContours(actin_cnt_mask, [actin_fiber.cnts[-1]], -1, 255, -1)

                        intersection = np.count_nonzero(cv2.bitwise_and(new_layer_cnt_mask, actin_cnt_mask))
                        if intersection > 0 and intersection > new_layer_cnt_obj.xsection:
                            new_layer_cnt_obj.xsection = intersection
                            new_layer_cnt_obj.parent = i

                # assign contour to actin fibers from previous layer
                for i, actin_fiber in enumerate(actin_fibers_from_previous_layer):
                    children_cnts = [new_layer_cnt for new_layer_cnt in actin_cnt_objs if new_layer_cnt.parent == i]

                    if len(children_cnts) == 1:
                        new_layer_cnt = children_cnts[0]
                        actin_fiber.update(new_layer_cnt.x, new_layer_cnt.y, new_layer_cnt.z, x_slice,
                                           new_layer_cnt.cnt)

                    if len(children_cnts) > 1:
                        max_intersection, idx = 0, -1
                        for j, child_cnt in enumerate(children_cnts):
                            if child_cnt.xsection > max_intersection:
                                max_intersection = child_cnt.xsection
                                idx = j

                        new_layer_cnt = children_cnts[idx]
                        actin_fiber.update(new_layer_cnt.x, new_layer_cnt.y, new_layer_cnt.z, x_slice,
                                           new_layer_cnt.cnt)

                        for j, child_cnt in enumerate(children_cnts):
                            if j != idx:
                                actin_fibers.append(
                                    SingleFiber(child_cnt.x, child_cnt.y, child_cnt.z, x_slice, child_cnt.cnt))

                # create new ActinFibers for contour objects which were not assigned to any existed ActinFibers
                for child_cnt in actin_cnt_objs:
                    if child_cnt.parent is None:
                        actin_fibers.append(SingleFiber(child_cnt.x, child_cnt.y, child_cnt.z, x_slice, child_cnt.cnt))
        actin_fibers_filtered = [fiber for fiber in actin_fibers if fiber.n >= fiber_min_layers_theshold]

        return actin_fibers_filtered

    def reconstruct(self, rot_angle, cnt_extremes, folders, unet_parm, part, fiber_min_layers_theshold, resolution):
        """
        Reconstruct fibers and saves all stat info into this Fibers objects
        ---
        Parameters:
        - rot_angle (int): rotation angle to rotate images of actin before making cross-section images for segmentation
        - rotated_cnt_extremes (CntExtremes object):
        - folders: (list): list of folders to save intermediate images during analysis
        - unet_parm (UnetParam object):
        - part:
            "whole" - analyze whole cell
            "cap" - analyze apic fibers
            "bottom" -  analyze basal fibers
        """

        print("\nFinding actin fibers...")
        # Step 1:
        actin_3d_img, rotated_max_projection = Utils.rotate_and_get_3D(folders["cut_out_nuc"], "actin", rot_angle)
        mid_cut_img, z_start, z_end = Utils.get_yz_xsection(actin_3d_img, folders["actin_xsection"], "actin",
                                                            cnt_extremes)
        # here as an option it is possibe to use different models for whole cell, top, bottom
        run_predict_unet(folders["actin_xsection"], folders["actin_mask"], unet_parm.actin_unet_model,
                         unet_parm.unet_model_scale,
                         unet_parm.unet_model_thrh)

        fibers_3D_mask = Utils.get_3d_img(folders["actin_mask"])
        actin_fibers = self._get_actin_fibers(fibers_3D_mask, fiber_min_layers_theshold)
        if part == "cap" or part == "bottom":
            for fiber in actin_fibers:
                fiber.assign_cap_or_bottom(z_start, z_end)
            filtered_actin_fibers = [fiber for fiber in actin_fibers if fiber.part == part]
            actin_fibers = filtered_actin_fibers
        self.fibers_list = actin_fibers
        self._add_fibers_aggregated_stat(resolution)

        return rotated_max_projection, mid_cut_img

    def plot(self, user_title=""):
        ax = plt.axes(projection='3d')
        if self.is_merged:
            for merged_fiber in self.fibers_list:
                color_x = 1.0 * np.random.randint(255) / 255
                color_y = 1.0 * np.random.randint(255) / 255
                color_z = 1.0 * np.random.randint(255) / 255
                for fiber in merged_fiber.fibers:
                    if len(np.unique(fiber.zs)) < 1:
                        continue
                    # Draw only center points
                    xdata = fiber.xs
                    ydata = fiber.ys
                    zdata = fiber.zs
                    if xdata:
                        ax.scatter3D(xdata, ydata, zdata, c=[[color_x, color_y, color_z]] * len(xdata), cmap='Greens')
            plt.title(f"Merged fibers \n {user_title}")
            plt.show()

        else:
            for fiber in self.fibers_list:
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
            plt.title(f"Single fibers \n {user_title}")
            plt.show()


    def find_connections(self, pyramid_apex_angle, fiber_joint_angle_z, max_distance, resolution):
        """
        Join fibers if, based on the analysis, fibers look like segments of one fiber:
        - candidate fiber's edge node (right_node) should be located within the pyramid projected from the main node (left_node)

                           /  +----- (first candidate)
                          /
        main node  ----+    (con_angle * 2)
                         \      +----- (second candidate)
                          \
        - The central ax of the pyramid is aligned according to the main fiber direction.
          To do so, we rotate the xy projection so the main fiber is parallel to the x-axes
        -------------------------                               -------------------------
        |         /  candidate 1 |                              |                 /      |
        |        /               |                              |               /        |
        |            -- -- --    |                              |             /          |
        |      ++    candidate 2 |    --> rotate based    -->   |   == == ==+    -- --   |  --> only candidate 1 is located within
        |     //                 |        on main (double)      |  main fiber \  can 1   |      the area of interest. H
        |    //                  |        line angle            |             |\         |      However, if we neglect rotation, candidate 2
        |   //                   |                              |             | \        |      will be assigned as a segment of the main fiber
        | main fiber             |                              |       can2  |          |
        --------------------------                              -------------------------

    Also we consider difference between xz and xy since xz has curve tendency
        ---
        Parameters:
        - pyramid_apex_angle (int): pyramid apex angle in degrees
        - max_distance (int): maximal distance between nodes, when two fibers can be considered as the pieces of one fiber
        - resolution: (ImgResolution object): x,y,z pixels sizes
        """
        self.fibers_list, nodes = add_edge_nodes(self.fibers_list)

        # Actin merging code (triangle approach), create merge candidate dictionary
        right_nodes = [(node_id, node) for node_id, node in enumerate(nodes)
                       if len(node.actin_ids) == 1 and self.fibers_list[node.actin_ids[0]].right_node_id == node_id]

        left_nodes = [(node_id, node) for node_id, node in enumerate(nodes)
                      if len(node.actin_ids) == 1 and self.fibers_list[node.actin_ids[0]].left_node_id == node_id]

        node_to_candidates = defaultdict(lambda: [])
        for righ_node_id, right_node in right_nodes:
            fiber = self.fibers_list[right_node.actin_ids[0]]
            rot_angle_xy = fiber.find_fiber_alignment_angle("xy")
            rot_angle_xz = fiber.find_fiber_alignment_angle("xz")
            anchor_point = (0, 0)
            right_rotated_xy = Utils.rotate_point((right_node.x,right_node.y), anchor_point, rot_angle_xy)
            right_rotated_xz = Utils.rotate_point((right_node.x, right_node.z), anchor_point, rot_angle_xz)
            for left_node_id, left_node in left_nodes:
                left_rotated_xy = Utils.rotate_point((left_node.x,left_node.y), anchor_point, rot_angle_xy)
                left_rotated_xz = Utils.rotate_point((left_node.x,left_node.z), anchor_point, rot_angle_xz)
                distance_between_nodes = (np.linalg.norm(np.array([left_node.x, left_node.y, left_node.z/(resolution.z/resolution.x)]) - np.array([right_node.x, right_node.y, right_node.z/(resolution.z/resolution.x)])))
                if right_rotated_xy[0] < left_rotated_xy[0] and distance_between_nodes <= max_distance:
                    if Utils.is_point_in_pyramid(right_rotated_xy, right_rotated_xz, left_rotated_xy, left_rotated_xz, pyramid_apex_angle, fiber_joint_angle_z, max_distance, resolution):
                        candidate_fiber = self.fibers_list[left_node.actin_ids[0]]
                        if candidate_fiber.n < distance_between_nodes or fiber.n < distance_between_nodes or abs(candidate_fiber.find_fiber_alignment_angle("xy")) > 60:
                            continue
                        node_to_candidates[righ_node_id].append([left_node_id, distance_between_nodes])

        node_to_candidates_list = []
        for k, v in node_to_candidates.items():
            node_to_candidates[k] = sorted(v, key=lambda x: x[1])
            node_to_candidates_list.append([k, v])

        # Create actual pairs of nodes to connect
        pairs = []
        while len(node_to_candidates_list) > 0:
            # find min distance
            min_distance = 10000
            r, l, pop_idx = None, None, None
            for idx, (right_node, candidate_distance_list) in enumerate(node_to_candidates_list):
                if min_distance > candidate_distance_list[0][1]:
                    min_distance = candidate_distance_list[0][1]
                    r, l, pop_idx = right_node, candidate_distance_list[0][0], idx
            pairs.append([r, l])
            node_to_candidates_list.pop(pop_idx)

            # remove left node from others candidates
            for right_node, candidate_distance_list in node_to_candidates_list:
                lefts = [item[0] for item in candidate_distance_list]
                if l in lefts:
                    candidate_distance_list.pop(lefts.index(l))

            indicies_to_remove = []
            for idx, (right_node, candidate_distance_list) in enumerate(node_to_candidates_list):
                if len(candidate_distance_list) == 0:
                    indicies_to_remove.append(idx)

            for idx in indicies_to_remove[::-1]:
                node_to_candidates_list.pop(idx)

        return nodes, pairs


    def find_connections_old_version(self, con_angle, min_len, resolution):
        """
        The previous version of connect function does not align the pyramid's
        central ax according to the main fiber direction.
        """
        self.fibers_list, nodes = add_edge_nodes(self.fibers_list)

        # Actin merging code (triangle approach), create merge candidate dictionary
        right_nodes = [(node_id, node) for node_id, node in enumerate(nodes)
                       if len(node.actin_ids) == 1 and self.fibers_list[node.actin_ids[0]].right_node_id == node_id]

        left_nodes = [(node_id, node) for node_id, node in enumerate(nodes)
                      if len(node.actin_ids) == 1 and self.fibers_list[node.actin_ids[0]].left_node_id == node_id]

        node_to_candidates = defaultdict(lambda: [])
        for righ_node_id, right_node in right_nodes:
            for left_node_id, left_node in left_nodes:
                if right_node.x < left_node.x and np.linalg.norm(np.array((right_node.x, right_node.y)) - np.array((left_node.x, left_node.y))) <= min_len:
                    if Utils.is_point_in_pyramid_old_version(right_node.x, right_node.y, right_node.z, left_node.x, left_node.y, left_node.z,
                                                     con_angle, min_len, resolution):
                            node_to_candidates[righ_node_id].append([left_node_id,
                                                                     np.sqrt((left_node.x - right_node.x) ** 2 +
                                                                             (left_node.y - right_node.y) ** 2 +
                                                                             (left_node.z - right_node.z) ** 2)])
        node_to_candidates_list = []
        for k, v in node_to_candidates.items():
            node_to_candidates[k] = sorted(v, key=lambda x: x[1])
            node_to_candidates_list.append([k, v])

        # Create actual pairs of nodes to connect
        pairs = []
        while len(node_to_candidates_list) > 0:
            # find min distance
            min_distance = 10000
            r, l, pop_idx = None, None, None
            for idx, (right_node, candidate_distance_list) in enumerate(node_to_candidates_list):
                if min_distance > candidate_distance_list[0][1]:
                    min_distance = candidate_distance_list[0][1]
                    r, l, pop_idx = right_node, candidate_distance_list[0][0], idx
            pairs.append([r, l])
            node_to_candidates_list.pop(pop_idx)

            # remove left node from others candidates
            for right_node, candidate_distance_list in node_to_candidates_list:
                lefts = [item[0] for item in candidate_distance_list]
                if l in lefts:
                    candidate_distance_list.pop(lefts.index(l))

            indicies_to_remove = []
            for idx, (right_node, candidate_distance_list) in enumerate(node_to_candidates_list):
                if len(candidate_distance_list) == 0:
                    indicies_to_remove.append(idx)

            for idx in indicies_to_remove[::-1]:
                node_to_candidates_list.pop(idx)

        return nodes, pairs

    def _add_fibers_aggregated_stat(self, resolution):
        total_volume = 0
        total_actin_length = 0
        total_num = 0
        slopes = []
        for i, fiber in enumerate(self.fibers_list):
            actin_length = (fiber.xs[-1] - fiber.xs[0]) * resolution.x
            actin_xsection = np.mean([cv2.contourArea(cnt) for cnt in fiber.cnts]) * resolution.y * resolution.z
            actin_volume = actin_length * actin_xsection
            total_actin_length = total_actin_length + actin_length
            total_volume = total_volume + actin_volume
            total_num = total_num + 1
            if fiber.xs[-1] - fiber.xs[0] == 0:
                slope = math.degrees(math.tan((fiber.ys[-1] - fiber.ys[0]) / 1))
            else:
                slope = math.degrees(math.tan((fiber.ys[-1] - fiber.ys[0]) / (fiber.xs[-1] - fiber.xs[0])))
            slopes.append(slope)
        if len(slopes) < 2: #Fix "statistics.StatisticsError: variance requires at least two data points"
            self.slope_variance = 0
        else:
            self.slope_variance = statistics.variance(slopes)
        self.total_volume = total_volume
        self.total_length = total_actin_length
        self.total_num = total_num


    def _update_merged_fibers_aggregated_stat(self, resolution):
        total_volume = 0
        total_actin_length = 0
        total_num = 0
        slopes = []
        for i, fiber in enumerate(self.fibers_list):
            merged_actin_length_with_gaps, merged_actin_xsection, merged_actin_volume_with_gaps, merged_n, slop = fiber.get_stat(resolution)
            total_actin_length = total_actin_length + merged_actin_length_with_gaps
            total_volume = total_volume + merged_actin_volume_with_gaps
            total_num = total_num + 1
            slopes.append(slop)
        if len(slopes) < 2: #Fix "statistics.StatisticsError: variance requires at least two data points"
            self.slope_variance = 0
        else:
            self.slope_variance = statistics.variance(slopes)

        self.total_volume = total_volume
        self.total_length = total_actin_length
        self.total_num = total_num


    def save_each_fiber_stat(self, resolution, file_path):
        header_row = ["ID", "Actin Length", "Actin Xsection", "Actin Volume", "Number of fiber layers", "Slope"]
        with open(file_path, mode='w') as stat_file:
            csv_writer = csv.writer(stat_file, delimiter=',')
            csv_writer.writerow(header_row)

            for fiber_id, fiber in enumerate(self.fibers_list):
                csv_writer.writerow([str(fiber_id)] + fiber.get_stat(resolution))


    def merge_fibers(self, fibers, nodes, pairs, resolution):
        merged_fibers = []

        for i, fiber in enumerate(fibers.fibers_list):
            merged_fiber = MergedFiber(fiber, i)
            merged_fibers.append(merged_fiber)

        for pair in pairs:
            left_node = nodes[pair[0]]
            right_node = nodes[pair[1]]
            left_fiber_id = left_node.actin_ids[0]
            right_fiber_id = right_node.actin_ids[0]
            merged_left_fiber = [merged_fiber for merged_fiber in merged_fibers if left_fiber_id in merged_fiber.fibers_ids][0]
            merged_fibers.remove(merged_left_fiber)
            merged_right_fiber = [merged_fiber for merged_fiber in merged_fibers if right_fiber_id in merged_fiber.fibers_ids][0]
            merged_fibers.remove(merged_right_fiber)
            merged_left_fiber.merge(merged_right_fiber, left_node, right_node, resolution)
            merged_fibers.append(merged_left_fiber)

        self.is_merged = True
        self.fibers_list = merged_fibers
        self._update_merged_fibers_aggregated_stat(resolution)















