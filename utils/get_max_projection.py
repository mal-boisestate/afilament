import time
import javabridge
import bioformats
import logging
import json
import os
import cv2.cv2 as cv2
from types import SimpleNamespace

from afilament.objects.CellAnalyser import CellAnalyser
from afilament.objects.CellAnalyser import Utils
from afilament.objects.ConfocalImgReader import ConfocalImgReader

def main():
    # Load JSON configuration file. This file can be produced by GUI in the future implementation
    with open("../afilament/config.json", "r") as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    img_nums = range(24)

    javabridge.start_vm(class_path=bioformats.JARS)
    analyser = CellAnalyser(config)
    start = time.time()
    logging.basicConfig(filename='../afilament/myapp.log', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)

    for img_num in img_nums:
        # analyser.save_nuc_verification(img_num)
        # To be able visually to verify intermediate steps the program keeps transitional images and all statistic data in the temp folder.
        raw_img_dir = r'C:\Users\nnina\Desktop\temp'
        output_dir = r'C:\Users\nnina\Desktop\img2'
        Utils.prepare_folder(raw_img_dir)

        reader = ConfocalImgReader(config.confocal_img, config.nucleus_channel, config.actin_channel, img_num)
        reader.read(raw_img_dir, "whole")

        img_base_path = os.path.splitext(os.path.basename(reader.image_path))[0]
        max_projection_origin_size, max_progection_unet_size = Utils.find_max_projection(raw_img_dir, "nucleus",
                                                                                         config.norm_th_nuc,
                                                                                         show_img=False)
        max_projection_path = os.path.join(output_dir, img_base_path + ".png")
        cv2.imwrite(max_projection_path, max_projection_origin_size)


    javabridge.kill_vm()


if __name__ == '__main__':
    main()