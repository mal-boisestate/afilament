from afilament.objects.Nucleus import Nucleus
from afilament.objects.Fibers import Fibers
from afilament.objects import Node


class Cell(object):
    def __init__(self, cell_num):
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
                             fiber_min_layers_theshold, resolution, is_plot_fibers):
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
        rotated_max_projection, mid_cut_img = fibers.reconstruct(rot_angle, rotated_cnt_extremes, folders,
                                                                 unet_parm, part, fiber_min_layers_theshold, resolution)
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
            self.actin_total_with_nodes, self.total_nodes = Node.run_node_creation(self.actin_total.fibers_list, new_actin_len_th, is_plot_nodes)
        elif part == "cap":
            self.actin_cap_with_nodes, self.cap_nodes = Node.run_node_creation(self.actin_cap.fibers_list, new_actin_len_th, is_plot_nodes)
        elif part == "bottom":
            self.actin_bottom_with_nodes, self.bottom_nodes = Node.run_node_creation(self.actin_bottom.fibers_list, new_actin_len_th, is_plot_nodes)

    def get_aggregated_cell_stat(self):
        """
        [_, "Cell_num", "Nucleus_volume, cubic_micrometre", "Nucleus_length, micrometre", "Nucleus_width, micrometre",
         "Nucleus_high, micrometre", "Total_fiber_num", "Cap_fiber_num", "Bottom_fiber_num", "Total_fiber_volume, cubic_micrometre",
         "Cap_fiber_volume, cubic_micrometre", "Bottom_fiber_volume, cubic_micrometre", "Total_fiber_length, micrometre",
         "Cap_fiber_length, micrometre", "Bottom_fiber_length, micrometre",
         "Slope_total_variance", "Slope_cap_variance", "Slope_bottom_variance",
         "Nodes_total, #", "Nodes_total, #", "Nodes_bottom, #"]
        """
        return [self.number, self.nucleus.nuc_volume, self.nucleus.nuc_length, self.nucleus.nuc_width, self.nucleus.nuc_high,
                self.actin_total.total_num, self.actin_cap.total_num, self.actin_bottom.total_num,
                self.actin_total.total_volume, self.actin_cap.total_volume, self.actin_bottom.total_volume,
                self.actin_total.total_length, self.actin_cap.total_length, self.actin_bottom.total_length,
                self.actin_total.slope_variance, self.actin_cap.slope_variance, self.actin_bottom.slope_variance,
                len(self.total_nodes), len(self.cap_nodes), len(self.bottom_nodes)]
