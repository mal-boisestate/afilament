import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from tqdm import tqdm

# === CONFIG ===
image_dir = r'C:\Users\nnina\Desktop\CNT_2nd_paper'  # directory with max projections (e.g., DAPI)
mask_dir = r'C:\Users\nnina\Desktop\mask_cnt_2nd_paper'    # directory with matching binary masks (same filenames)
output_csv = r'C:\Users\nnina\Desktop\cnt_ccp_results.csv'

# === Normalize image by clipping at 99.5% and scaling to 0–255 ===
def normalize_image(img, upper_percentile=99.5):
    img = img.astype(np.float32)
    upper = np.percentile(img, upper_percentile)
    img_clipped = np.clip(img, 0, upper)
    img_normalized = (img_clipped / upper) * 255
    return img_normalized.astype(np.uint8)

# === CCP calculation function ===
def calculate_ccp(image, mask):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)

    threshold = np.percentile(sobel_mag, 90)
    edge_mask = (sobel_mag > threshold) & (mask > 0)

    edge_pixels = np.count_nonzero(edge_mask)
    nucleus_area = np.count_nonzero(mask)

    return edge_pixels / nucleus_area if nucleus_area > 0 else 0

# === Main processing loop ===
results = []

file_list = [f for f in os.listdir(image_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))]

for filename in tqdm(file_list):
    image_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Skipping {filename} - missing image or mask")
        continue

    # === Normalize the image BEFORE segmentation ===
    image = normalize_image(image, upper_percentile=99.5)

    labeled_mask = label(mask > 0, connectivity=1)
    props = regionprops(labeled_mask)

    for prop in props:
        nucleus_id = prop.label
        nucleus_mask = (labeled_mask == nucleus_id).astype(np.uint8)

        ccp = calculate_ccp(image, nucleus_mask)

        results.append({
            'Image': filename,
            'Nucleus_ID': nucleus_id,
            'CCP': ccp
        })

# === Save results ===
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"\n✅ Results saved to {output_csv}")
