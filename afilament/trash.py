import glob
import os
import numpy as np
import cv2.cv2 as cv2
import pathlib
from afilament.objects import Utils

input_folder = r"C:\Users\nnina\Desktop\actin_new_img"
output_folder = r"C:\Users\nnina\Desktop\actin_new_img_8_bit"
input_path = pathlib.Path(input_folder)

Utils.save_as_8bit(input_folder, output_folder)


# fibers_3D_img = Utils.get_3d_img(r"D:\BioLab\scr_2.0\afilament\temp\actin_layers")
# one_slice = fibers_3D_img[150, :, :265]
# sum_one_slice = np.sum(one_slice)
# cv2.imshow("one slice", one_slice)
# cv2.waitKey(0)
# check_total = np.sum(fibers_3D_img, dtype=np.int64)
#
# b = np.sum(fibers_3D_img[:, :, :265], dtype=np.int64)
# c = np.sum(fibers_3D_img[:, :, 265:], dtype=np.int64)
#
# a = 1


