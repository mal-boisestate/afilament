from afilament.objects.Nucleus import Nucleus
from afilament.objects.Fibers import Fibers
from afilament.objects import Node
from afilament.objects.Parameters import TestStructure
import pickle
import math


class Cell(object):
    def __init__(self, img_num, cell_num):
        self.img_number = img_num
        self.number = cell_num
        self.nucleus = None
        self.actin_total = None
        self.actin_cap = None
        self.actin_bottom = None
        self.actin_total_with_nodes = None
        self.actin_cap_with_nodes = None
        self.actin_bottom_with_nodes = None
        self.total_nodes = None
        self.cap_nodes = None
        self.bottom_nodes = None
        self.actin_total_join = None
        self.actin_cap_join = None
        self.actin_bottom_join = None


    def analyze_nucleus(self, rot_angle, rotated_cnt_extremes, folders, unet_parm, img_resolution,
                        analysis_folder, norm_th_nuc):
        """
        Run nucleus analysis of the cell
        ---
        Parameters:
        - rot_angle (int): rotation angle to rotate images of the nucleus before making cross-section images for segmentation
        - rotated_cnt_extremes (CntExtremes object):
        - folders: (list): list of folders to save intermediate images during analysis
        - unet_parm (UnetParam object):
        img_resolution (ImgResolution object):
        """
        nucleus = Nucleus()
        nucleus.reconstruct(rot_angle, rotated_cnt_extremes, folders, unet_parm, img_resolution,
                            analysis_folder, norm_th_nuc)
        self.nucleus = nucleus


    def analyze_actin_fibers(self, rot_angle, rotated_cnt_extremes, folders, unet_parm, part,
                             fiber_min_thr_microns, resolution, is_plot_fibers, is_connect_fibers,
                             fiber_joint_angle, fiber_joint_distance, cap_bottom_ratio, norm_th_actin):
        """
        Run fibers analysis based on part and save results in cell (self) object
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
        fibers = Fibers(part)
        cap_bottom_cut_off_z = self.nucleus.get_cut_off_z(cap_bottom_ratio)


        rotated_max_projection, mid_cut_img = fibers.reconstruct(rot_angle, rotated_cnt_extremes, folders,
                                                                 unet_parm, part, resolution, cap_bottom_cut_off_z,
                                                                 norm_th_actin)
        if is_connect_fibers:
            fiber_joint_angle_z = math.degrees(math.atan(self.nucleus.nuc_length/(self.nucleus.nuc_high/2))) #Fiber join angle for z axis is calculated based on nucleus height and length so the algorithm considers curve surface of nucleus
            nodes, pairs = fibers.find_connections(fiber_joint_angle, fiber_joint_angle_z, fiber_joint_distance, resolution)
            nodes_old, pairs_old = fibers.find_connections_old_version(fiber_joint_angle, fiber_joint_distance, resolution)
            if is_connect_fibers:
                print(f"New approach connection num: {len(pairs)}")
                print(f"Old approach connection num: {len(pairs_old)}")
                Node.plot_connected_nodes(fibers.fibers_list, nodes, pairs, "New method", False)
                Node.plot_connected_nodes(fibers.fibers_list, nodes_old, pairs_old, "Old method", False)
                test_structure = TestStructure(fibers, nodes, pairs, resolution)

                with open('test_structure.pickle', "wb") as file_to_save:
                    pickle.dump(test_structure, file_to_save)

                join_fibers = Fibers(part)
                join_fibers.merge_fibers(fibers, nodes, pairs, resolution)

                join_fibers_old = Fibers(part)
                join_fibers_old.merge_fibers(fibers, nodes_old, pairs_old, resolution)

                if part == "whole":
                    self.actin_total_join = join_fibers
                elif part == "bottom":
                    self.actin_bottom_join = join_fibers
                elif part == "cap":
                    self.actin_cap_join = join_fibers
                if is_plot_fibers:
                    join_fibers.plot("New method")
                    join_fibers_old.plot("Old method")

        if part == "whole":
            self.actin_total = fibers
        elif part == "bottom":
            self.actin_bottom = fibers
        elif part == "cap":
            self.actin_cap = fibers
        else:
            raise ValueError(f"argument part should be whole, cap, or bottom, but {part} is specified")
        if is_plot_fibers:
            fibers.plot(fiber_min_thr_microns)
        return rotated_max_projection, mid_cut_img


    def find_branching(self, fiber_min_thr_microns, new_actin_len_th):
        if self.actin_total:
            actin_fibers_filtered = [fiber for fiber in self.actin_total.fibers_list if fiber.length >= fiber_min_thr_microns]
            self.total_nodes, self.actin_total_with_nodes = Node.find_branching_nodes(actin_fibers_filtered,
                                                                                      new_actin_len_th)
        if self.actin_cap:
            actin_fibers_filtered = [fiber for fiber in self.actin_cap.fibers_list if fiber.length >= fiber_min_thr_microns]
            self.cap_nodes, self.actin_cap_with_nodes = Node.find_branching_nodes(actin_fibers_filtered,
                                                                                  new_actin_len_th)
        if self.actin_bottom:
            actin_fibers_filtered = [fiber for fiber in self.actin_bottom.fibers_list if fiber.length >= fiber_min_thr_microns]
            self.bottom_nodes, self.actin_bottom_with_nodes = Node.find_branching_nodes(actin_fibers_filtered,
                                                                                        new_actin_len_th)

    def get_aggregated_cell_stat(self, is_separate_cap_bottom, fiber_min_thr_microns, resolution, node_actin_len_th):
        """
        [_, "Img_num", "Cell_num", "Nucleus_volume, cubic_micrometre", "Nucleus_length, micrometre",
        "Nucleus_width, micrometre", "Nucleus_high, micrometre",
        "Nucleus_total_intensity", "Total_fiber_num", "Cap_fiber_num", "Bottom_fiber_num",
        "Total_fiber_volume, cubic_micrometre", "Cap_fiber_volume, cubic_micrometre",
        "Bottom_fiber_volume, cubic_micrometre", "Total_fiber_length, micrometre",
         "Cap_fiber_length, micrometre", "Bottom_fiber_length, micrometre",
         "Fiber_intensity_whole", "Fiber_intensity_cap", "Fiber_intensity_bottom",
         "F-actin_signal_intensity_whole", "F-actin_signal_intensity_cap", "F-actin_signal_intensity_bottom",
         "Branching_nodes_total, #", "Branching_nodes_cap, #", "Branching_nodes_bottom, #"]
        """

        self.find_branching(fiber_min_thr_microns, node_actin_len_th)

        if is_separate_cap_bottom:
            self.actin_total.create_fibers_aggregated_stat(fiber_min_thr_microns, resolution)
            self.actin_cap.create_fibers_aggregated_stat(fiber_min_thr_microns, resolution)
            self.actin_bottom.create_fibers_aggregated_stat(fiber_min_thr_microns, resolution)
            total_branching_nodes = [node for node in self.total_nodes if len(node.actin_ids) > 1]
            cap_branching_nodes = [node for node in self.cap_nodes if len(node.actin_ids) > 1]
            bottom_branching_nodes = [node for node in self.bottom_nodes if len(node.actin_ids) > 1]

            return [self.img_number, self.number, self.nucleus.nuc_volume, self.nucleus.nuc_length,
                    self.nucleus.nuc_width, self.nucleus.nuc_high_alternative,
                    self.nucleus.nuc_intensity,
                    self.actin_total.total_num, self.actin_cap.total_num, self.actin_bottom.total_num,
                    self.actin_total.total_volume, self.actin_cap.total_volume, self.actin_bottom.total_volume,
                    self.actin_total.total_length, self.actin_cap.total_length, self.actin_bottom.total_length,
                    self.actin_total.intensity, self.actin_cap.intensity, self.actin_bottom.intensity,
                    self.actin_total.f_actin_signal_total_intensity, self.actin_cap.f_actin_signal_total_intensity,
                    self.actin_bottom.f_actin_signal_total_intensity,
                    len(total_branching_nodes), len(cap_branching_nodes), len(bottom_branching_nodes)]
        else:
            self.actin_total.create_fibers_aggregated_stat(fiber_min_thr_microns, resolution)
            total_branching_nodes = [node for node in self.total_nodes if len(node.actin_ids) > 1]
            actin_total_alternative_length = self.actin_total.get_alternative_length_test_k(fiber_min_thr_microns,
                                                                                            resolution, step=5)
            return [self.img_number, self.number, self.nucleus.nuc_volume, self.nucleus.nuc_length,
                    self.nucleus.nuc_width, self.nucleus.nuc_high_alternative,
                    self.nucleus.nuc_intensity, self.actin_total.total_num, self.actin_total.total_volume,
                    self.actin_total.total_length,
                    self.actin_total.intensity, self.actin_total.f_actin_signal_total_intensity, len(total_branching_nodes)]

    def update_actin_stat_old_format(self, resolution):
        """
        This function is a temporary solution to revert the calculation of action length to the old format, specifically
        focusing on changes along the x-axis and disregarding changes across the y and z-axes. The decision to adopt
        this method was made during a meeting with Dr. Uzer on 05/26/2023. The alternative methods of calculating length
        have raised numerous questions, and we have chosen to allocate more time to investigate these concerns.
        """

        def update_fiber_length_and_av_xsection(fiber, resolution):
            fiber.length = (fiber.xs[-1] - fiber.xs[0]) * resolution.x

            if fiber.length == 0:
                fiber.av_xsection = 0
            else:
                fiber.av_xsection = fiber.volume / fiber.length

        for fiber_list in [getattr(self.actin_total, 'fibers_list', None), getattr(self.actin_cap, 'fibers_list', None), getattr(self.actin_bottom, 'fibers_list', None)]:
            if fiber_list is not None:
                for fiber in fiber_list:
                    update_fiber_length_and_av_xsection(fiber, resolution)


