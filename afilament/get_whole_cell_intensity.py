import time
import javabridge
import bioformats
import logging
import os
import cv2
import numpy as np
import csv
import pandas as pd

from objects.ConfocalImgReader import ConfocalImgReader
from objects import Utils

def main():


    javabridge.start_vm(class_path=bioformats.JARS)
    start = time.time()
    logging.basicConfig(filename='myapp.log', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)


    mask_path = r"D:\BioLab\Current_experiments\2005.04.17_mitahondria_analysis\Masks"
    czi_images_path = r"Z:\Common\BMMB\LSM900 files\Uzer Lab\NN\Images\2025.04.10_Mitahondria_mice\stiched"
    temp = r'D:\BioLab\Current_experiments\2005.04.17_mitahondria_analysis\temp'
    verification_dir = r'D:\BioLab\Current_experiments\2005.04.17_mitahondria_analysis\verification'

    # --- Load mask and find contours ---
    def find_contours(mask_file):
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, mask.shape

    # --- Get image list from temp (after nucleus images removed) ---
    def load_images(temp_folder):
        image_stack = []
        image_names = []

        for f in sorted(os.listdir(temp_folder)):
            if "nucleus" in f.lower():
                os.remove(os.path.join(temp_folder, f))
                continue

            img = cv2.imread(os.path.join(temp_folder, f), cv2.IMREAD_UNCHANGED)  # 16-bit
            if img is not None:
                image_stack.append(img)
                image_names.append(f)

        return image_stack, image_names

    # --- Analyze contour intensity ---
    def analyze_contours(contours, images):
        results = []

        for i, contour in enumerate(contours):
            mask = np.zeros_like(images[0], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)  # filled contour

            area = cv2.countNonZero(mask)
            sums = [int(np.sum(img[mask == 255])) for img in images]

            results.append({
                "contour_index": i,
                "pixel_count": area,
                "intensity_sums": sums
            })

        return results

    # --- Create and save verification image ---
    def save_verification(contours, images, save_path):
        max_proj = np.max(np.stack(images), axis=0)

        # Normalize to 3000 and convert to 8-bit
        norm = np.clip((max_proj / 3000) * 255, 0, 255).astype(np.uint8)
        norm_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

        # Draw contours in random colors
        for i, contour in enumerate(contours):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.drawContours(norm_rgb, [contour], -1, color, 1)

        cv2.imwrite(save_path, norm_rgb)

    # --- Main execution loop ---
    # Get filenames without extensions
    mask_files = {os.path.splitext(f)[0]: os.path.join(mask_path, f)
                  for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f))}
    czi_files = {os.path.splitext(f)[0]: os.path.join(czi_images_path, f)
                 for f in os.listdir(czi_images_path) if os.path.isfile(os.path.join(czi_images_path, f))}

    common_keys = set(mask_files.keys()).intersection(czi_files.keys())

    for img_num in sorted(common_keys):
        mask_file = mask_files[img_num]
        czi_file = czi_files[img_num]

        # Read and export images into temp using your reader
        reader = ConfocalImgReader(czi_file, nucleus_channel=1, actin_channel=0, cell_number=0)
        Utils.prepare_folder(temp)
        reader.read(temp)

        # Load non-nucleus images
        images, img_names = load_images(temp)

        if not images:
            print(f"No valid images found in {temp} for {img_num}")
            continue

        # Find contours
        contours, mask_shape = find_contours(mask_file)

        # Analyze
        data = analyze_contours(contours, images)

        # Save verification image
        verification_path = os.path.join(verification_dir, f"{img_num}_verification.png")
        save_verification(contours, images, verification_path)

        # Save to CSV
        csv_path = os.path.join(verification_dir, f"{img_num}_contour_data.csv")
        with open(csv_path, mode='w', newline='') as csv_file:
            fieldnames = ['image_id', 'contour_index', 'pixel_count'] + img_names
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for d in data:
                row = {
                    'image_id': img_num,
                    'contour_index': d['contour_index'],
                    'pixel_count': d['pixel_count']
                }
                row.update({name: intensity for name, intensity in zip(img_names, d['intensity_sums'])})
                writer.writerow(row)

        print(f"Saved CSV: {csv_path}")

    combined_csv_path = os.path.join(verification_path, "combined_results.csv")

    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(verification_path) if f.endswith(".csv")]

    # Combine all into one DataFrame
    combined_df = pd.concat(
        [pd.read_csv(os.path.join(verification_path, f)) for f in csv_files],
        ignore_index=True
    )

    # Save to one CSV
    combined_df.to_csv(combined_csv_path, index=False)

    print(f"Combined CSV saved to: {combined_csv_path}")


    end = time.time()
    print("Total time is: ")
    print(end - start)
    javabridge.kill_vm()


if __name__ == '__main__':
    main()