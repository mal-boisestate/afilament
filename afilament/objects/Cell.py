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


    def analyze_nucleus(self, rot_angle, rotated_cnt_extremes, folders, unet_parm, img_resolution, analysis_folder):
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
        nucleus.reconstruct(rot_angle, rotated_cnt_extremes, folders, unet_parm, img_resolution, analysis_folder)
        self.nucleus = nucleus


    def analyze_actin_fibers(self, rot_angle, rotated_cnt_extremes, folders, unet_parm, part,
                             fiber_min_layers_theshold, resolution, is_plot_fibers, is_connect_fibers,
                             fiber_joint_angle, fiber_joint_distance, cap_bottom_ratio):
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
                                                                 unet_parm, part, fiber_min_layers_theshold,
                                                                 resolution, cap_bottom_cut_off_z)
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
            fibers.plot()
        return rotated_max_projection, mid_cut_img


    def find_branching(self, part, new_actin_len_th, is_plot_nodes):
        if part == "whole":
            self.actin_total_with_nodes, self.total_nodes = Node.find_branching_nodes(self.actin_total.fibers_list, new_actin_len_th, is_plot_nodes)
        elif part == "cap":
            self.actin_cap_with_nodes, self.cap_nodes = Node.find_branching_nodes(self.actin_cap.fibers_list, new_actin_len_th, is_plot_nodes)
        elif part == "bottom":
            self.actin_bottom_with_nodes, self.bottom_nodes = Node.find_branching_nodes(self.actin_bottom.fibers_list, new_actin_len_th, is_plot_nodes)

    def get_aggregated_cell_stat(self, is_separate_cap_bottom):
        """
        [_, "Img_num", "Cell_num", "Nucleus_volume, cubic_micrometre", "Nucleus_length, micrometre", "Nucleus_width, micrometre",
         "Nucleus_high, micrometre", "Nucleus_high_alternative, micrometre", "Nucleus_total_intensity",", "Total_fiber_num", "Cap_fiber_num", "Bottom_fiber_num", "Total_fiber_volume, cubic_micrometre",
         "Cap_fiber_volume, cubic_micrometre", "Bottom_fiber_volume, cubic_micrometre", "Total_fiber_length, micrometre",
         "Cap_fiber_length, micrometre", "Bottom_fiber_length, micrometre",
         "Fiber_intensity_whole", "Fiber_intensity_cap", "Fiber_intensity_bottom",
         "F-actin_signal_intensity_whole", "F-actin_signal_intensity_cap", "F-actin_signal_intensity_bottom",
         "Nodes_total, #", "Nodes_total, #", "Nodes_bottom, #"]
        """
        if is_separate_cap_bottom:
            return [self.img_number, self.number, self.nucleus.nuc_volume, self.nucleus.nuc_length,
                    self.nucleus.nuc_width, self.nucleus.nuc_high, self.nucleus.nuc_high_alternative, self.nucleus.nuc_intensity,
                    self.actin_total.total_num, self.actin_cap.total_num, self.actin_bottom.total_num,
                    self.actin_total.total_volume, self.actin_cap.total_volume, self.actin_bottom.total_volume,
                    self.actin_total.total_length, self.actin_cap.total_length, self.actin_bottom.total_length,
                    self.actin_total.intensity, self.actin_cap.intensity, self.actin_bottom.intensity,
                    self.actin_total.f_actin_signal_total_intensity, self.actin_cap.f_actin_signal_total_intensity,
                    self.actin_bottom.f_actin_signal_total_intensity,
                    len(self.total_nodes), len(self.cap_nodes), len(self.bottom_nodes)]
        else:
            return [self.img_number, self.number, self.nucleus.nuc_volume, self.nucleus.nuc_length,
                    self.nucleus.nuc_width, self.nucleus.nuc_high, self.nucleus.nuc_high_alternative,
                    self.nucleus.nuc_intensity, self.actin_total.total_num, self.actin_total.total_volume, self.actin_total.total_length,
                    self.actin_total.intensity, self.actin_total.f_actin_signal_total_intensity, len(self.total_nodes)]



