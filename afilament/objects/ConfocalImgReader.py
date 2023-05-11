import os
import sys
import cv2.cv2 as cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import math
import bioformats

from afilament.objects.Parameters import ImgResolution


# TODO this class shuld be changed, so imagies for czi old formst, czi new format and lif can be read automatically
# TODO now there is problem in def _save_img() since all theree formats have different array dota for the image
# TODO lif has multiple imagies for file. CZI only one. So for czi folder should be passed, for lif - file.
# TODO now code is adapted for czi
# I decided to do different readers - one for czi and another for lif since they are too different

class ConfocalImgReaderCzi(object):
    """
    Creates an object that reads confocal microscopy images of two channels (actin and nucleus)
    """

    def __init__(self, folder_path, nucleus_channel, actin_channel):
        """
            Parameters:
            img_path (string): path to the file to read
            nucleus_channel(int): channel of nucleus images at the provided microscopic image
            actin_channel(int): channel of actin images at the provided microscopic image
        """
        self.folder_path = folder_path
        self.img_name = None
        self.nuclei_channel = nucleus_channel
        self.actin_channel = actin_channel
        self.type, self.image_arrays = None, None

    def _read_as_czi(self, image_num, part):
        img_path = None
        for i, path in enumerate(Path(self.folder_path).rglob('*.czi')):
            if i == image_num:
                self.img_name = path.name
                img_path = path
                break
        with CziFile(img_path) as czi:
            self.image_arrays = czi.asarray()
        self.type = self.image_arrays[0, 0, 0, :, :, 0].dtype  # this for zen withoud lasers
        edge = math.ceil(self.image_arrays.shape[2] * 1 / 2)  # image_arrays.shape[2] is z stack for apotome czi
        if part == "cap":
            self.image_arrays = self.image_arrays[:, :, 0:edge, :, :, :]
        elif part == "bottom":
            self.image_arrays = self.image_arrays[:, :, edge:self.image_arrays.shape[2] - 1, :, :, :]

    # this function shuld be modifyed based on file type, for czi and lif index
    def _save_img(self, bio_structure, output_folder, pixel_value=256):
        for i in tqdm(range(self.image_arrays.shape[2])):  # for confocal czi index shuld be 4
            img_path = self._get_img_path(self.img_name + "_" + bio_structure, i, output_folder)
            channel = self.actin_channel if bio_structure == "actin" else self.nuclei_channel
            # cv2.imwrite(img_path, np.uint8(self.image_arrays[:, :, i, channel] / 1))  # convert to 8-bit image
            img = self.image_arrays[0, channel, i, :, :, 0]  # czi appotome
            # img = self.image_arrays[0, 0, channel, 0, i, :, :, 0] #czi confocal
            img_8bit = np.uint8(img / (pixel_value / 256))
            cv2.imwrite(img_path, img_8bit)  # convert to 8-bit image. This conversio is very strange. nut it works

    def _get_img_path(self, img_name, layer, output_folder):
        img_path_norm = os.path.join(output_folder, img_name + '_layer_' + str(layer) + '.png')
        return img_path_norm

    def read(self, output_folder, image_num, norm_trh, part="whole"):
        """
        Converts confocal microscopic images into a set of jpg images specified in reader object normalization
        ---
            Parameters:
            output_folder (string): path to the folder to save jpg images
            modifier part:
                "whole" - analyze whole cell
                "cap" - analyze 2/3 upper part of the cell that we consider as an actin cap
                "bottom" -  analyze 1/3 of the bottom part the we consider as a bottom
        """
        self._read_as_czi(image_num, part)
        self._normalize_and_save_img("actin", output_folder, norm_trh)
        self._normalize_and_save_img("nucleus", output_folder, norm_trh)

    def find_norm_thr(self, cell_nums):
        # Check average threshold for the first three cells, if there are less then 3 cells check for all
        nuc_cut_offs = []
        actin_cut_offs = []

        for cell_num in cell_nums:
            # if len(cell_nums) <= i:
            #     break
            self._read_as_czi(cell_num, "whole")
            edge = math.ceil(self.image_arrays.shape[2] * 1 / 2)  # image_arrays.shape[2] is z stack for apotome czi
            nuc_img = self.image_arrays[0, self.nuclei_channel, edge - 1:edge, :, :, 0][0, :, :]
            actin_img = self.image_arrays[0, self.actin_channel, edge - 1:edge, :, :, 0][0, :, :]
            # Utils.plot_histogram(f"Middle layer of Cell {cell_num} nucleus", nuc_img)
            # Utils.plot_histogram(f"Middle layer of Cell {cell_num} actin", actin_img)
            share = 0.05 / 100  # pixel intensity that we are interested into
            index = int(share * nuc_img.shape[0] * nuc_img.shape[1])
            nuc_cut_off = np.sort(nuc_img.flatten())[-index]
            actin_cut_off = np.sort(actin_img.flatten())[-index]
            nuc_cut_offs.append(nuc_cut_off)
            actin_cut_offs.append(actin_cut_off)
        print(f"Nuc cut off is {nuc_cut_offs}")
        print(f"Nuc cut off is {actin_cut_offs}")
        return (int(np.max(nuc_cut_offs)), int(np.max(actin_cut_offs)))

    def _normalize_and_save_img(self, bio_structure, output_folder, norm_thr):
        pixel_value = 65536 if self.type == "uint16" else 256

        for i in tqdm(range(self.image_arrays.shape[2])):  # for confocal czi index shuld be 4
            img_path = self._get_img_path(self.img_name + "_" + bio_structure, i, output_folder)
            channel = self.actin_channel if bio_structure == "actin" else self.nuclei_channel
            img = self.image_arrays[0, channel, i, :, :, 0]  # czi appotome
            if norm_thr is None:
                img_8bit = np.uint8(img / (pixel_value / 256))
            else:
                norm_th_bio_structure = norm_thr[1] if bio_structure == "actin" else norm_thr[0]
                img_8bit = self.normalization(img, norm_th_bio_structure)

            cv2.imwrite(img_path, img_8bit)  # convert to 8-bit image. This conversio is very strange. nut it works

    def normalization(self, img, norm_th):
        img[np.where(img > norm_th)] = norm_th
        img = cv2.normalize(img, None, alpha=0, beta=norm_th, norm_type=cv2.NORM_MINMAX)
        img = (img / 256).astype(np.uint8)

        return img


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
        Converts confocal microscopic images into a set of png images specified in reader object normalization
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
            """I am trying to redo it to actualy image size, for example 16 bits,
            so I can have more inensity data - remove that commnets after story will be done"""
            # img = self.normalization(img, type)


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

    def normalization(self, img, type):
        img[np.where(img > self.norm_th)] = self.norm_th
        # img = cv2.normalize(img, None, alpha=0, beta=self.norm_th, norm_type=cv2.NORM_MINMAX)

        if type == 'uint16':
            img = (img / (self.norm_th / math.pow(2, 8))).astype(np.uint8)

        elif type == 'uint12':
            img = (img / (self.norm_th / math.pow(2, 4))).astype(np.uint8)

        return img