import os
from PIL import Image
import cv2.cv2 as cv2
import numpy as np
from afilament.objects import Contour
from unet.predict import run_predict_unet, run_predict_unet_one_img


output_folder_path = r"C:\Users\nnina\Desktop\dirty_masks"
output_folder_path_final = r"C:\Users\nnina\Desktop\nuc_mask_unet"
nuc_area_min_pixels_num = 20000
run_predict_unet(r"C:\Users\nnina\Desktop\2", output_folder_path,
                 "unet/models/top_nuc_models/CP_epoch100_max_pr_186cells_incl_cis.pth",
                 1,
                 0.5)

for filename in os.listdir(output_folder_path):
    file_path = os.path.join(output_folder_path, filename)
    mask_img = Image.open(file_path)
    #To_Do convert mask_img to np array
    mask_img = np.array(mask_img)

    kernel_dil = np.ones((5, 5), np.uint8)
    kernel_er = np.ones((5, 5), np.uint8)

    img_dilation = cv2.dilate(mask_img, kernel_dil, iterations=1)
    img_erotion = cv2.erode(img_dilation, kernel_er, iterations=1)
    cnts = Contour.get_img_cnts(img_erotion, theshold=100)
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > nuc_area_min_pixels_num]  # removes noise
    dim = mask_img.shape

    processed_nuc_mask = np.zeros(dim, dtype="uint8")
    cv2.drawContours(processed_nuc_mask, cnts, -1, color=(255, 255, 255), thickness=cv2.FILLED)
    one_nuc_mask_path = os.path.join(output_folder_path_final, filename)

    cv2.imwrite(one_nuc_mask_path, processed_nuc_mask)





