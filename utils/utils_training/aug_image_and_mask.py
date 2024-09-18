import os
import numpy as np
from random import randrange
import random
from PIL import Image, ImageOps
import glob


def ensure_8bpp(image):
    """
    Convert image to 8bpp grayscale if it is 1bpp.
    """
    if image.mode == '1':  # 1bpp image (binary)
        image = image.convert('L')  # Convert to 8bpp grayscale
    elif image.mode != 'L':  # Ensure the image is in 8bpp grayscale
        image = image.convert('L')
    return image


if __name__ == '__main__':
    folder_path = r"D:\BioLab\scr_2.0\unet\data\imgs"
    mask_folder_path = r"D:\BioLab\scr_2.0\unet\data\masks"

    # Find all image files (.png and .bmp)
    image_files = glob.glob(os.path.join(folder_path, "*.png")) + glob.glob(os.path.join(folder_path, "*.bmp"))
    mask_files = glob.glob(os.path.join(mask_folder_path, "*.png")) + glob.glob(os.path.join(mask_folder_path, "*.bmp"))

    # Convert mask file names (without extensions) to a dictionary for faster lookup
    mask_dict = {os.path.splitext(os.path.basename(mask))[0]: mask for mask in mask_files}

    for img_path in image_files:
        img_file = os.path.basename(img_path)
        img_name, img_ext = os.path.splitext(img_file)  # Get image name without extension

        # Find corresponding mask based on the image name (regardless of extension)
        if img_name not in mask_dict:
            print(f"No mask found for {img_file}, skipping.")
            continue

        mask_path = mask_dict[img_name]

        # Open the image and mask files
        nucleus_img = Image.open(img_path)
        nucleus_mask = Image.open(mask_path)

        # Convert mask to 8bpp if necessary
        nucleus_mask = ensure_8bpp(nucleus_mask)

        # Get image dimensions
        width, height = nucleus_img.size
        width_shift = 1 / 2 * (width - ((height / 2) ** 2 + (width / 2) ** 2) ** (1 / 2))
        height_shift = 1 / 2 * (height - ((height / 2) ** 2 + (width / 2) ** 2) ** (1 / 2))

        # Gamma correction options
        gamma = [0.80, 0.90, 1, 1, 1, 1.10, 1.20]

        # Augmentation loop
        for j in range(3):
            # Apply gamma correction and rotate
            gamma_corrected = Image.fromarray(
                np.array(255 * (np.array(nucleus_img) / 255) ** random.choice(gamma), dtype='uint8'))
            angle_to_rotate = random.choice([0, 90, 180, 270])
            aug_img = gamma_corrected.rotate(angle_to_rotate)
            aug_mask = nucleus_mask.rotate(angle_to_rotate)

            # Random flipping
            if random.choice([1, 0]):
                aug_img = ImageOps.flip(aug_img)
                aug_mask = ImageOps.flip(aug_mask)
            if random.choice([1, 0]):
                aug_img = ImageOps.mirror(aug_img)
                aug_mask = ImageOps.mirror(aug_mask)

            # Save augmented images
            img_file_name = '_aug_' + str(j) + '_' + img_name + '.png'
            mask_file_name = '_aug_' + str(j) + '_' + img_name + '.png'  # Ensure mask name matches image name
            img_path_to_save = os.path.join(r"D:\BioLab\scr_2.0\unet\data\imgs_aug", img_file_name)
            mask_path_to_save = os.path.join(r"D:\BioLab\scr_2.0\unet\data\masks_aug", mask_file_name)

            # Save image and mask as PNG
            aug_img.save(img_path_to_save, "PNG")
            aug_mask.save(mask_path_to_save, "PNG")
            print(f"Processing {img_file} and {mask_file_name}")
