import os
import sys
import cv2.cv2 as cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import math
import bioformats

from afilament.objects.Parameters import ImgResolution


class ConfocalImgReader(object):
    """
    Creates an object that reads confocal microscopy images of two channels (actin and nucleus)
    """

    def __init__(self, path, nucleus_channel, actin_channel, cell_number, norm_th):
        """
            Parameters:
            img_path (string): path to the file to read
            nucleus_channel(int): channel of nucleus images at the provided microscopic image
            actin_channel(int): channel of actin images at the provided microscopic image
        """
        self.image_path, self.series = self.get_img_path_and_series(path, cell_number)
        self.actin_channel = actin_channel
        self.nuc_channel = nucleus_channel
        self.norm_th = norm_th
        metadata = bioformats.get_omexml_metadata(str(self.image_path))
        self.metadata_obj = bioformats.OMEXML(metadata)
        self.update_channels()
        self.img_resolution = self.get_resolution()

    def get_resolution(self):
        scale_x = self.metadata_obj.image(self.series).Pixels.get_PhysicalSizeX()
        scale_y = self.metadata_obj.image(self.series).Pixels.get_PhysicalSizeY()
        scale_z = self.metadata_obj.image(self.series).Pixels.get_PhysicalSizeZ()
        img_resolution = ImgResolution(scale_x, scale_y, scale_z)
        return img_resolution

    def get_img_path_and_series(self, path, cell_number):
        """
        CZI and LIF files, in our case organized differently.
        LIF is a project file that has different images as a Series.
        CZI is a path to the folder that contains separate images.
        This method checks what is the case and finds the path-specific image and Series.
        Args:
            path: str, path to folder or project file

        Returns:
            img_path: path to file
            series: series to analyze
        """
        img_path = None
        if os.path.isfile(path):
            series = cell_number
            img_path = path

        else:
            series = 0
            folder_path = path
            for i, current_path in enumerate(Path(folder_path).rglob('*.czi')):
                if i == cell_number:
                    img_path = current_path
                    break
        print(img_path)

        return img_path, series

    def update_channels(self):
        """
        Update channels based on channel names "DAPI" for nucleus and "AF488" for actin.
        Print error if it is not possible to update channels and Channels were not specified by the user.
        """
        for i in range(self.metadata_obj.image().Pixels.get_channel_count()):
            if self.metadata_obj.image().Pixels.Channel(i).get_Name() == "DAPI":
                self.nuc_channel = i
            elif self.metadata_obj.image().Pixels.Channel(i).get_Name() == "AF488":
                self.actin_channel = i
        if self.actin_channel is None or self.nuc_channel is None:
            sys.exit("Please specify channels for actin and nucleus. "
                     "This data can not be extracted from the metadata file")

    def read(self, output_folder, part="whole"):
        """
        Converts confocal microscopic images into a set of png images specified in reader object
        ---
            Parameters:
            output_folder (string): path to the folder to save jpg images
            modifier part:
                "whole" - analyze whole cell
                "cap" - analyze 1/2 upper part of the cell that we consider as an actin cap
                "bottom" -  analyze 1/2 of the bottom part the we consider as a bottom
        """



        nucleus_img_stack = self._get_img_stack(bio_structure="nucleus")
        actin_img_stack = self._get_img_stack(bio_structure="actin")

        z_layers_num = len(nucleus_img_stack)
        edge = math.ceil(z_layers_num * 1 / 2)  # image_arrays.shape[2] is z stack for apotome czi

        if part == "cap":
            nucleus_img_stack = nucleus_img_stack[0:edge]
            actin_img_stack = actin_img_stack[0:edge]
        elif part == "bottom":
            nucleus_img_stack = nucleus_img_stack[edge:z_layers_num-1]
            actin_img_stack = actin_img_stack[edge:z_layers_num-1]
        self._save_img(nucleus_img_stack, "nucleus", output_folder)
        self._save_img(actin_img_stack, "actin", output_folder)



    def _get_img_stack(self, bio_structure):
        channel = self.actin_channel if bio_structure == "actin" else self.nuc_channel
        image_stack = []
        z_layers_num = self.metadata_obj.image(self.series).Pixels.get_SizeZ()
        for i in range(z_layers_num):
            img = bioformats.load_image(str(self.image_path), c=channel, z=i, t=0, series=self.series, index=None, rescale=False,
                                        wants_max_intensity=False,
                                        channel_names=None)

            type = self.metadata_obj.image(self.series).Pixels.get_PixelType()

            image_stack.append(img)
        return image_stack

    def _save_img(self, img_stack, bio_structure, output_folder):
        for i in tqdm(range(len(img_stack))):
            img_name = os.path.splitext(os.path.basename(self.image_path))[0]
            img_path = os.path.join(output_folder,
                                    img_name + '_series_' + str(self.series) +'_' + bio_structure + '_layer_' + str(i) +'.png')
            cv2.imwrite(img_path, img_stack[i])

    def close(self):
        bioformats.clear_image_reader_cache()
