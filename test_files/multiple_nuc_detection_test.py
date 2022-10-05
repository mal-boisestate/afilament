from afilament.objects.Utils import find_max_projection, prepare_folder, normalization
from afilament.objects.ConfocalImgReader import ConfocalImgReader
import os
import cv2.cv2 as cv2
import javabridge
import bioformats
from unet.predict import run_predict_unet


#1. Read all imagies from the folder in the np.array
#2. Create 3D np.array
#3. Create max proojection and see results
# input_folder_czi = r'D:\BioLab\img\Chase_img\20x\Replicate 3 (6-7-22)\Control'
input_folder_czi = r"D:\BioLab\img\Confocal_img\2022.09.21_DAPI_488\2022.09.21_LIV"
raw_img_folder = r"D:\BioLab\Current_experiments\cut_multiple_cells\raw_img_folder"
max_projection_folder = r"D:\BioLab\Current_experiments\cut_multiple_cells\max_projection_folder"
mask_folder = r"D:\BioLab\Current_experiments\cut_multiple_cells\mask_folder"
nucleus_channel = 1
actin_channel = 0
norm_th = 2**16 / 3

from_top_nucleus_unet_model = r"D:\BioLab\scr_2.0\unet\models\CP_epoch200_max_pr.pth"
# from_top_nucleus_unet_model = r"D:\BioLab\models\one_nucleus_top_view\CP_epoch200_nucleus_weight.corection_2_labling_512_512_with_agum.pth"
unet_model_scale = 1
unet_img_size = (512, 512)
unet_model_thrh = 0.5


identifier = "nucleus"
javabridge.start_vm(class_path=bioformats.JARS)





for i in range(20):
    prepare_folder(raw_img_folder)
    reader = ConfocalImgReader(input_folder_czi, nucleus_channel, actin_channel, i, norm_th)
    reader.read(raw_img_folder, 'whole')

    max_projection, max_progection_img = find_max_projection(raw_img_folder, identifier, show_img=False)
    max_projection_init_file_path = os.path.join(max_projection_folder, os.path.basename(str(reader.image_path)) + "_" + str(i) + "_max_projection.png")
    cv2.imwrite(max_projection_init_file_path, max_projection)

run_predict_unet(max_projection_folder, mask_folder, from_top_nucleus_unet_model, unet_model_scale, unet_model_thrh)

javabridge.kill_vm()
#4. Recognize max projection imagies using unet - see results
    #4.1 Save image in specify folder
    #4.2 Run unet and see results (choose 5 imagies)
#5. Recognize max projection using threshold but change size from 512*512 to actual size

#5. Hot to choose the right nuclei??? Circular, not cutted ect.