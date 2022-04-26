from afilament.objects.Nucleus import Nucleus
from afilament.objects.Fibers import Fibers


class Cell(object):
    def __init__(self, cell_num):
        self.number = cell_num
        self.nucleus = None
        self.actin_total = None
        self.actin_cap = None
        self.actin_bottom = None


    def analyze_nucleus(self, rot_angle, rotated_cnt_extremes, folders, unet_parm, img_resolution):
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
        nucleus.reconstruct(rot_angle, rotated_cnt_extremes, folders, unet_parm, img_resolution)
        self.nucleus = nucleus


    def analyze_actin_fibers(self, rot_angle, rotated_cnt_extremes, folders, unet_parm, part, fiber_min_layers_theshold, resolution):
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

        return rotated_max_projection, mid_cut_img


    def get_aggregated_cell_stat(self):
        """
        [_, "Cell_num", "Nucleus_volume, cubic_micrometre", "Nucleus_length, micrometre", "Nucleus_width, micrometre",
         "Nucleus_high, micrometre", "Total_fiber_num", "Cap_fiber_num", "Bottom_fiber_num", "Total_fiber_volume, cubic_micrometre",
         "Cap_fiber_volume, cubic_micrometre", "Bottom_fiber_volume, cubic_micrometre", "Total_fiber_length, micrometre",
         "Cap_fiber_length, micrometre", "Bottom_fiber_length, micrometre"]
        """
        return [self.number, self.nucleus.nuc_volume, self.nucleus.nuc_length, self.nucleus.nuc_width, self.nucleus.nuc_high,
                self.actin_total.total_num, self.actin_cap.total_num, self.actin_bottom.total_num,
                self.actin_total.total_volume, self.actin_cap.total_volume, self.actin_bottom.total_volume,
                self.actin_total.total_length, self.actin_cap.total_length, self.actin_bottom.total_length]
