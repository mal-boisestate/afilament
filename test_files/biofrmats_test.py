import javabridge
import bioformats
import numpy as np
from pathlib import Path

import os
import cv2.cv2 as cv2

if __name__ == '__main__':
    confocal_img_path = r"D:\BioLab\img\Confocal_img\3D_new_set\2022.01.12_KASH_dox.lif"
    # confocal_img_path = r"D:\BioLab\img\Confocal_img\2022.05.10_DAPI_488\Experiment-118-ApoTome deconvolution-41.czi"
    path = r"D:\BioLab\img\Confocal_img\2022.05.10_DAPI_488"


    javabridge.start_vm(class_path=bioformats.JARS)
    for i, current_path in enumerate(Path(path).rglob('*.czi')):
        print(current_path)


    metadata = bioformats.get_omexml_metadata(confocal_img_path)
    o = bioformats.OMEXML(metadata)


    x = o.image().Pixels.get_PhysicalSizeX()
    y = o.image().Pixels.get_PhysicalSizeY()
    z = o.image().Pixels.get_PhysicalSizeZ()
    type = o.image().Pixels.get_PixelType()
    channels_names = o.image().Pixels.Channel(0).get_Name()
    channels_count = o.image().Pixels.get_channel_count()
    actin_channel = None
    nuc_channel = None
    for i in range(o.image().Pixels.get_channel_count()):
        if o.image().Pixels.Channel(i).get_Name() == "DAPI":
            nuc_channel = i
        elif o.image().Pixels.Channel(i).get_Name() == "AF488":
            actin_channel = i
    if actin_channel == None or nuc_channel == None:
        print("Please specify channels for actin and nucleus. This data can not be extracted from metadata file")
        #then put channel names so we can use them later when loading imagies
        actin_channel = 0
        nuc_channel = 1

    z_layers_num = o.image(1).Pixels.get_SizeZ()
    output_folder = r"D:\BioLab\Current_experiments\bioformat_python_reader_experimetns"
    img_name = "test"
    for i in range(z_layers_num):
        img = bioformats.load_image(confocal_img_path, c=nuc_channel, z=i, t=0, series=1, index=None,
                                    rescale=False,
                                    wants_max_intensity=False,
                                    channel_names=None)

        img_path = os.path.join(output_folder, img_name + '_layer_' + str(i) + '.png')

        pixel_value = 256
        cv2.imwrite(img_path, img)

    bioformats.clear_image_reader_cache()
    javabridge.kill_vm()

