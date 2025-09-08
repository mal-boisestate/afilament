import os
import cv2
import numpy as np

# Define input and output folders
folder_a = r'C:\Users\nnina\Desktop\3'
folder_b = r'C:\Users\nnina\Desktop\2'
output_folder = r'C:\Users\nnina\Desktop\combined'

os.makedirs(output_folder, exist_ok=True)

# List of filenames assuming both folders have matching filenames
file_names_a = set(os.listdir(folder_a))
file_names_b = set(os.listdir(folder_b))
file_names = file_names_a.intersection(file_names_b)

# Process and combine images
for file in file_names:
    # Load grayscale images
    img_a = cv2.imread(os.path.join(folder_a, file), cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(os.path.join(folder_b, file), cv2.IMREAD_GRAYSCALE)

    # Check if images loaded correctly
    if img_a is None or img_b is None:
        print(f"Skipping {file}, unable to load images.")
        continue

    # Normalize images to range [0,1]
    img_a_norm = cv2.normalize(img_a, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    img_b_norm = cv2.normalize(img_b, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # Create color images: 'a' as orange, 'b' as blue
    color_a = np.zeros((*img_a.shape, 3), dtype=np.float32)
    color_a[:, :, 0] = img_a_norm * 0      # Blue channel
    color_a[:, :, 1] = img_a_norm * 0.5    # Green channel
    color_a[:, :, 2] = img_a_norm * 1      # Red channel (orange = red + half green)

    color_b = np.zeros((*img_b.shape, 3), dtype=np.float32)
    color_b[:, :, 0] = img_b_norm * 1      # Blue channel
    color_b[:, :, 1] = img_b_norm * 0      # Green channel
    color_b[:, :, 2] = img_b_norm * 0      # Red channel

    # Combine images by overlaying
    combined = np.clip(color_a + color_b, 0, 1)

    # Convert back to 8-bit
    combined_img = (combined * 255).astype(np.uint8)

    # Save result
    output_path = os.path.join(output_folder, file)
    cv2.imwrite(output_path, combined_img)

print("All images processed and saved.")
