import pickle
import os
import csv
import json
from datetime import datetime

from afilament.objects import Utils
from afilament.objects.ConfocalImgReader import ConfocalImgReader
from afilament.objects import Contour
from afilament.objects.Cell import Cell
from afilament.objects.Parameters import UnetParam

temp_folders = {
    "raw": '../afilament/temp/czi_layers',
    "cut_out_nuc": '../afilament/temp/actin_and_nucleus_layers',
    "actin_xsection": '../afilament/temp/actin_layers',
    "nucleous_xsection": '../afilament/temp/nucleus_layers',
    "actin_mask": '../afilament/temp/actin_mask',
    "nucleus_mask": '../afilament/temp/nucleus_mask',
    "nucleus_top_mask": '../afilament/temp/nucleus_top_mask',
    "nucleus_top_img": '../afilament/temp/nucleus_top_img',
    "nuclei_top_masks": '../afilament/temp/nuclei_top_masks',
}

analysis_data_folders = {
    "area_ver": '../afilament/analysis_data/nuc_area_verification',
    "mesh": '../afilament/analysis_data/mesh',
    "rotatation_verification": '../afilament/analysis_data/rotatation_verification',
    "actin_stat": '../afilament/analysis_data/actin_stat',
    "actin_objects": '../afilament/analysis_data/actin_objects',
    "middle_xsection": '../afilament/analysis_data/middle_xsection',
    "analysis": '../afilament/analysis_data/analysis'
}


class CellAnalyser(object):
    def __init__(self, config):
        self.initial_conf = config
        self.nucleus_channel = config.nucleus_channel
        self.actin_channel = config.actin_channel
        self.confocal_path = config.confocal_img
        self.nuc_theshold = config.nuc_theshold
        self.unet_parm = UnetParam(config)
        self.fiber_min_layers_theshold = config.fiber_min_layers_theshold
        self.node_actin_len_th = config.node_actin_len_th
        self.is_plot_fibers = config.is_plot_fibers
        self.is_plot_nodes = config.is_plot_nodes
        self.norm_th = config.norm_th
        self.find_biggest_mode = config.find_biggest_mode #"unet" for U-Net mode or "trh" for trh mode
        self.img_resolution = None
        self.is_separate_cap_bottom = config.is_separate_cap_bottom
        self.is_connect_fibers = config.is_connect_fibers
        self.fiber_joint_angle = config.fiber_joint_angle
        self.fiber_joint_distance = config.fiber_joint_distance
        self.nuc_area_min_pixels_num = config.nuc_area_min_pixels_num
        self.cap_bottom_ratio = config.cap_bottom_ratio
        self.is_auto_normalize = config.is_auto_normalized
        self.total_img_number = 0
        self.total_cells_number = 0


        for folder in analysis_data_folders.values():
            Utils.prepare_folder(folder)

    def analyze_cell(self, img_num, cell_num, mask, reader):
        """
        Run analysis of specified cell part, update information in cell and return it
    ---
        Parameters:
        - cell (Cell object): cell to analyse
        - cap = True if apical fibers should be analysed separately False otherwise
        - bottom = True if basal fibers should be analysed separately False otherwise

        """
        cell = Cell(img_num, cell_num)
        self._run_analysis(cell, "whole", mask, reader)
        if self.is_separate_cap_bottom:
            self._run_analysis(cell, "cap", mask, reader)
            self._run_analysis(cell, "bottom", mask, reader)

        return cell


    def save_nuc_verification(self, img_num):
        """
        Save nucleus area verificatin imagies. This function is helpful to verify different settings
        """
        for folder in temp_folders.values():
            Utils.prepare_folder(folder)

        reader = ConfocalImgReader(self.confocal_path, self.nucleus_channel, self.actin_channel, img_num, self.norm_th)
        reader.read(temp_folders["raw"], "whole")
        Utils.get_nuclei_masks(temp_folders, analysis_data_folders,
                               reader.image_path, self.nuc_theshold, self.nuc_area_min_pixels_num,
                               self.find_biggest_mode, img_num, self.unet_parm)

    def analyze_img(self, img_num):
        """
        Run analysis of the image. Finds all nuclei on the image
        """

        # For metadata statistics
        self.total_img_number += 1

        # To be able visually to verify intermediate steps the program keeps transitional images and all statistic data in the temp folder.
        for folder in temp_folders.values():
            Utils.prepare_folder(folder)

        # Step 1: Read confocal microscope image, save images in png 8 bit. Since there are
        # computational power limitations our Unet works only with 8-bit images.
        reader = ConfocalImgReader(self.confocal_path, self.nucleus_channel, self.actin_channel, img_num, self.norm_th)
        self.img_resolution = reader.img_resolution
        reader.read(temp_folders["raw"], "whole")
        nuclei_masks = Utils.get_nuclei_masks(temp_folders, analysis_data_folders,
                                             reader.image_path, self.nuc_theshold, self.nuc_area_min_pixels_num,
                                             self.find_biggest_mode, img_num, self.unet_parm)

        cells = []
        for i, nuc_mask in enumerate(nuclei_masks):
            cell = self.analyze_cell(img_num, i, nuc_mask, reader)
            cells.append(cell)
            self.total_cells_number += 1
        return cells

    def _run_analysis(self, cell, part, nucleus_mask, reader):
        """
        General logic of cell analysis step by step is represented by this method:
        Step 1: Read confocal microscope image, save images in png 8 bit.
        Step 2: Go through all confocal image slices and find a slice with the biggest nucleus area.
        Step 3: Cut out this area from other slices.
        Step 4: Find rotation angle:
        - "whole" cell based on all layers
        - "cap" on upper half
        - "bottom" on lower "half"
        Step 5: Reconstruct the nucleus.
        Step 6: Reconstruct the specified area (whole/cap/bottom) of actin fibers
        ---
        Parameters:
        - cell (Cell object): cell to analyse
        - part:
            "whole" - analyze whole cell
            "cap" - analyze apic fibers
            "bottom" -  analyze basal fibers
        - biggest_nucleus_mask: an optional parameter, used with "cap" and "bottom" modifiers, ensures
                                that the analyzed (cut out) area for all modifiers is the same

        """
        # To be able visually to verify intermediate steps the program keeps transitional images and all statistic data in the temp folder.
        for folder in temp_folders.values():
            Utils.prepare_folder(folder)

        print(f"\n Analyse {part} fibers of the cell # {cell.number}")
        # Step 1: Read confocal microscope image, save images in png 8 bit. Since there are
        # computational power limitations our Unet works only with 8-bit images.
        # reader = ConfocalImgReader(self.confocal_path, self.nucleus_channel, self.actin_channel, cell.number, self.norm_th)
        self.img_resolution = reader.img_resolution
        reader.read(temp_folders["raw"], part)


        print("\nGenerate xsection images...")
        # Step 2: Cut out this nucleus mask area from all image slices.
        if part == "whole":
            Utils.сut_out_mask(nucleus_mask, temp_folders["raw"], temp_folders["cut_out_nuc"], 'nucleus')  # reconstruct nucleus based on "whole" cell
        Utils.сut_out_mask(nucleus_mask, temp_folders["raw"], temp_folders["cut_out_nuc"], 'actin')

        # Step 3: Find rotation angle and keep produced by rotation algorithm images (max_progection_img, hough_lines_img)
        # for further visual verification.
        # Finding the rotation angle for "whole" cell based on all layers, for "cap" on upper half, for "bottom" on lower "half"
        # Rotate mask, so area of interest can be catted again after rotation all layers.
        rot_angle, max_progection_img, hough_lines_img = Utils.find_rotation_angle(temp_folders["cut_out_nuc"])
        rotated_mask = Utils.rotate_bound(nucleus_mask, rot_angle)
        rotated_cnt = Contour.get_mask_cnt(rotated_mask)
        rotated_cnt_extremes = Contour.get_cnt_extremes(rotated_cnt)

        # Step 4: Reconstruct the nucleus. Run reconstruction only for the "whole" cell.
        # For "cap" and "bottom" we need only fibers.
        if part == "whole":
            cell.analyze_nucleus(rot_angle, rotated_cnt_extremes, temp_folders, self.unet_parm, self.img_resolution, analysis_data_folders["analysis"])

        # Step 5:  Reconstruct the specified area (whole/cap/bottom) of actin fibers:
        #  a. For "cap" and "bottom": rotate the whole nucleus again.
        #     It will help us better reconstruct the area of interest since "cap" and "bottom" fibers are not parallel,
        #     and this algorithm requires specific fibers crosssection cut for better reconstruction.
        #     We still need to run an analysis of all layers. It helps us not to cut fiber in the middle.
        #     But decide if fiber is "cap" or "bottom" based on the mean of z  of each individual fiber.
        if part == "cap" or part == "bottom":
            Utils.prepare_folder(temp_folders["raw"])
            Utils.prepare_folder(temp_folders["cut_out_nuc"])
            reader.read(temp_folders["raw"], "whole")
            Utils.сut_out_mask(nucleus_mask, temp_folders["raw"], temp_folders["cut_out_nuc"], 'actin')
            length = (rotated_cnt_extremes.right[0] - rotated_cnt_extremes.left[0]) * self.img_resolution.x
        rotated_max_projection, mid_cut_img = cell.analyze_actin_fibers(rot_angle, rotated_cnt_extremes, temp_folders,
                                                                        self.unet_parm, part, self.fiber_min_layers_theshold,
                                                                        self.img_resolution, self.is_plot_fibers,
                                                                        self.is_connect_fibers, self.fiber_joint_angle,
                                                                        self.fiber_joint_distance, self.cap_bottom_ratio)
        cell.find_branching(part, self.node_actin_len_th, self.is_plot_nodes)
        Utils.save_rotation_verification(cell, max_progection_img, hough_lines_img, rotated_max_projection, mid_cut_img,
                                         part, analysis_data_folders)
        reader.close()
        return nucleus_mask

    def save_cells_data(self, cells, cap=False, bottom=False):
        """
        Save whole fiber statistics for each cell in separate file. Save actin object as well.
        """
        for cell in cells:
            # save actin fiber statistics
            actin_stat_total_file_path = os.path.join(analysis_data_folders["actin_stat"],
                                                      "img_num_" + str(cell.img_number) + "__cell_num_" + str(cell.number)
                                                      + "_whole_actin_stat.csv")
            cell.actin_total.save_each_fiber_stat(self.img_resolution, actin_stat_total_file_path)
            actin_object_total = os.path.join(analysis_data_folders["actin_objects"],
                                              "img_num_" + str(cell.img_number) + "__cell_num_" + str(cell.number)
                                              + "_whole_3d_actin.obj")
            with open(actin_object_total, "wb") as file_to_save:
                pickle.dump(cell.actin_total, file_to_save)

            if self.is_separate_cap_bottom:
                actin_stat_cap_file_path = os.path.join(analysis_data_folders["actin_stat"],
                                                        "img_num_" + str(cell.img_number) + "__cell_num_" + str(cell.number)
                                                        + "_cap_actin_stat.csv")
                cell.actin_cap.save_each_fiber_stat(self.img_resolution, actin_stat_cap_file_path)
                actin_object_cap = os.path.join(analysis_data_folders["actin_objects"],
                                                "img_num_" + str(cell.img_number) + "__cell_num_" + str(cell.number)
                                                + "_cap_3d_actin.obj")
                with open(actin_object_cap, "wb") as file_to_save:
                    pickle.dump(cell.actin_cap, file_to_save)

                actin_stat_bottom_file_path = os.path.join(analysis_data_folders["actin_stat"],
                                                           "img_num_" + str(cell.img_number) + "__cell_num_" + str(cell.number)
                                                           + "_bottom_actin_stat.csv")
                cell.actin_bottom.save_each_fiber_stat(self.img_resolution, actin_stat_bottom_file_path)
                actin_object_bottom = os.path.join(analysis_data_folders["actin_objects"],
                                                   "img_num_" + str(cell.img_number) + "__cell_num_" + str(cell.number)
                                                   + "_bottom_3d_actin.obj")
                with open(actin_object_bottom, "wb") as file_to_save:
                    pickle.dump(cell.actin_bottom, file_to_save)


    def save_aggregated_cells_stat(self, cells):
        """
        Save aggregated statistical data in the file specified in folders["agreg_stat"] list
        Each raw of this file represent a cell, colums names are:
        - "Image_name"
        - "Image_number"
        - "Cell_num"
        - "Nucleus_volume, cubic_micrometre"
        - "Nucleus_length, micrometre",
        - "Nucleus_width, micrometre",
        - "Nucleus_high, micrometre"
        - "Total_fiber_num", "Cap_fiber_num"
        - "Bottom_fiber_num",
        - "Total_fiber_volume, cubic_micrometre"
        - "Cap_fiber_volume, cubic_micrometre"
        - "Bottom_fiber_volume, cubic_micrometre"
        - "Total_fiber_length, micrometre"
        - "Cap_fiber_length, micrometre"
        - "Bottom_fiber_length, micrometre"
        """
        if self.is_separate_cap_bottom:
            header_row = ["Image_name", "Img_num", "Cell_num", "Nucleus_volume, cubic_micrometre",
                          "Nucleus_length, micrometre", "Nucleus_width, micrometre",
                          "Nucleus_high, micrometre", "Total_fiber_num", "Cap_fiber_num", "Bottom_fiber_num",
                          "Total_fiber_volume, cubic_micrometre",
                          "Cap_fiber_volume, cubic_micrometre", "Bottom_fiber_volume, cubic_micrometre",
                          "Total_fiber_length, micrometre",
                          "Cap_fiber_length, micrometre", "Bottom_fiber_length, micrometre",
                          "Slope_total_variance",
                          "Slope_cap_variance", "Slope_bottom_variance",
                           "Nodes_total, #", "Nodes_total, #", "Nodes_bottom, #"
                          ]
            path = os.path.join(analysis_data_folders["analysis"], 'cell_stat.csv')
            with open(path, mode='w') as stat_file:
                csv_writer = csv.writer(stat_file, delimiter=',')
                csv_writer.writerow(header_row)
                for cell in cells:
                    csv_writer.writerow([str(self.confocal_path)] + cell.get_aggregated_cell_stat(self.is_separate_cap_bottom))
        else:
            header_row = ["Image_name","Img_num", "Cell_num", "Nucleus_volume, cubic_micrometre", "Nucleus_length, micrometre",
                          "Nucleus_width, micrometre",
                          "Nucleus_high, micrometre", "Total_fiber_num",
                          "Total_fiber_volume, cubic_micrometre", "Total_fiber_length, micrometre",
                          "Slope_total_variance","Nodes_total, #"]
            path = os.path.join(analysis_data_folders["analysis"], 'cell_stat.csv')
            with open(path, mode='w') as stat_file:
                csv_writer = csv.writer(stat_file, delimiter=',')
                csv_writer.writerow(header_row)
                for cell in cells:
                    csv_writer.writerow([str(self.confocal_path)] + cell.get_aggregated_cell_stat(self.is_separate_cap_bottom))

        print("Stat created")

    def save_config(self):
        # Add additional information
        analysis_date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        self.initial_conf.analysis_date_time = analysis_date
        self.initial_conf.total_img_number = self.total_img_number
        self.initial_conf.total_cells_number = self.total_cells_number
        # Serializing json
        json_conf_str = json.dumps(self.initial_conf, indent=4, default=lambda o: o.__dict__,
            sort_keys=True)

        # Writing to sample.json
        file_path = os.path.join(analysis_data_folders["analysis"], "analysis_configurations.json")
        with open(file_path, "w") as outfile:
            outfile.write(json_conf_str)





