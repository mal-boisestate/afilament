import glob
import os
import cv2.cv2 as cv2
import math
import sys
from afilament.objects import Contour
import numpy as np
import matplotlib.pyplot as plt
from unet.predict import run_predict_unet


def prepare_folder(folder):
    """
    Create folder if it has not been created before
    or clean the folder
    ---
    Parameters:
    -   folder (string): folder's path
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    for f in glob.glob(folder + "/*"):
        os.remove(f)


def find_max_projection(input_folder, identifier, show_img=False):
    object_layers = []
    for img_path in glob.glob(os.path.join(input_folder, "*_" + identifier + "_*.png")):
        layer = int(img_path.rsplit(".", 1)[0].rsplit("_", 1)[1])
        object_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        object_layers.append([object_img, layer])

    object_layers = sorted(object_layers, key=lambda x: x[1], reverse=True)
    image_3d = np.asarray([img for img, layer in object_layers], dtype=np.uint8)
    max_projection_origin_size = image_3d[:, :, :].max(axis=0, out=None, keepdims=False, where=True)
    max_progection_unet_size = cv2.resize(max_projection_origin_size, (512, 512))
    if show_img:
        cv2.imshow("Max projection", max_progection_unet_size)
        cv2.waitKey()
    return max_projection_origin_size, max_progection_unet_size


def find_rotation_angle(input_folder, rotation_trh=50):
    """
    Find rotation angle based on Hough lines of edges of maximum actin projection.
    ---
    Parameters:
        - input_folder (string): folder's path
        - rotation_trh (int): threshold for canny edges
    ___
    Returns:
        - rot_angle (int): rotaion angle in degrees
        - max_progection_img (img): image for verification
        - hough_lines_img (img): image for verification
    """
    identifier = "actin"
    max_projection, max_progection_img = find_max_projection(input_folder, identifier)

    # Find the edges in the image using canny detector
    edges = cv2.Canny(max_projection, rotation_trh, 100)

    # Detect points that form a line
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=50, maxLineGap=250)

    # Draw lines on the image
    angles = []
    if lines is None:
        rot_angle = 0
        hough_lines_img = np.zeros((1000, 1000), dtype=np.uint8)

    else:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(max_projection, (x1, y1), (x2, y2), (255, 0, 0), 3)
            if x1 == x2:
                angles.append(-90)
            else:
                angles.append(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
        # Create window with freedom of dimensions
        rot_angle = (np.median(angles))
        hough_lines_img = cv2.resize(max_projection, (1000, 1000))
        # cv2.imshow("output", cv2.resize(max_progection,(1000, 1000))) #keep it for debugging
        # cv2.waitKey()

    return -rot_angle, max_progection_img, hough_lines_img


def rotate_bound(image, angle):
    """
    Rotate provided image to specified angle
        ---
    Parameters:
        - image (img): image to rotate
        - angle (int): angle to rotate
    ___
    Returns:
        - rotated image
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def rotate_and_get_3D(input_folder, identifier, rot_angle):
    """
    Take identified images (actin/nucleus) with an already cut area of interest,
    rotate these images to a specified angle, and return a 3D image (stack of these 2D images).
    ---
    Parameters:
        - input_folder (string): path to input folder
        - identifier (string): actin or nucleus
        - rot_angle (int): angle to rotate the images to
    Returns:
        - image_3d (snp.array): rotated 3D image
        - max_progection_img (img): image for verification
    """
    object_layers = []
    for img_path in glob.glob(os.path.join(input_folder, "*_" + identifier + "_*.png")):
        layer = int(img_path.rsplit(".", 1)[0].rsplit("_", 1)[1])
        object_img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        object_img = rotate_bound(object_img, rot_angle)
        object_layers.append([object_img, layer])
        # cv2.imshow("output", cv2.resize(object_img, (1000, 1000))) #keep for debugging purposes
        # cv2.waitKey()
    object_layers = sorted(object_layers, key=lambda x: x[1], reverse=True)
    image_3d = np.asarray([img for img, layer in object_layers])
    max_progection = image_3d[:, :, :].max(axis=0, out=None, keepdims=False, where=True)
    max_progection_img = cv2.resize(max_progection, (512, 512))
    image_3d = np.moveaxis(image_3d, 0, -1)  # Before moving axis (z, x, y), after moving axis (x, y, z)

    return image_3d, max_progection_img


def save_as_8bit(input_folder, output_folder):
    for img_path in glob.glob(os.path.join(input_folder, "*.png")):
        object_img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        pixel_value = 65536 if object_img.dtype == "uint16" else 256
        img_8bit = np.uint8(object_img / (pixel_value / 256))
        output_img_path = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(output_img_path, img_8bit)


def get_yz_xsection(img_3d, output_folder, identifier, cnt_extremes, unet_img_size=512):
    """
    Save jpg cross-section of img_3d with padding in output_folder.
    ---
        Parameters:
        - img_3d (np. array): the three-dimensional array that represents 3D image of the identifier (nucleus/actin)
        - output_folder (string): path to the folder to save processed jpg images
        - identifier (string) "actin" or "nucleus"
        - cnt_extremes (CntExtremes object) where left, right, top, bottom attributes are coordinates
                                    of the corresponding extreme points of the biggest nucleus contour
    ---
        Returns:
        - mid_cut (img): for verification
        - x_xsection_start (img): for verification
        - x_section_end (img): for verification
    """
    top, bottom = cnt_extremes.top[1], cnt_extremes.bottom[1]
    x_start, x_end, step = cnt_extremes.left[0], cnt_extremes.right[0], 1
    x_middle = x_start + (x_end - x_start) // 2
    mid_cut = None  # saved later for verification
    x_xsection_start, x_section_end = 0, 0
    for x_slice in range(x_start, x_end, step):
        xsection = img_3d[top: bottom, x_slice, :]
        img = xsection
        padded_img, x_xsection_start, x_section_end = make_padding(img, unet_img_size)
        img_path = os.path.join(output_folder, "xsection_" + identifier + "_" + str(x_slice) + ".png")
        cv2.imwrite(img_path, padded_img)
        if x_slice == x_middle:
            mid_cut = padded_img
    return mid_cut, x_xsection_start, x_section_end


def make_padding(img, final_img_size):

    h, w = img.shape[:2]

    if max([h, w]) > final_img_size:
        final_img_size = max([h, w])

    h_out = w_out = final_img_size

    top = (h_out - h) // 2
    bottom = h_out - h - top
    left = (w_out - w) // 2
    right = w_out - w - left

    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded_img, left, (padded_img.shape[0] - right)


def get_3d_img(input_folder):
    """
    Reads and combines images from input_folder into image_3d according to layer number.
    ---
        Parameters:
        - input_folder (string): path to the input folder with jpg images
    ---
        Returns:
        - image_3d (np.array): three-dimensional array of images combined layer by layer together
    """
    object_layers = []

    for img_path in glob.glob(input_folder + r"\*"):
        img_name, img_ext = os.path.splitext(os.path.basename(img_path))
        layer = int(img_name.rsplit("_", 1)[1])  # layer number is part of the image name

        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        # img = np.flip(img, axis=0)
        object_layers.append([img, layer])

    object_layers = sorted(object_layers, key=lambda x: x[1], reverse=True)
    img_3d = np.asarray([mask for mask, layer in object_layers])

    return img_3d


def save_rotation_verification(cell, max_progection_img, hough_lines_img, rotated_max_projection, mid_cut_img, part,
                               folders):
    """
    Save images in specify folder
    ---
    """
    max_projection_init_file_path = os.path.join(folders["rotatation_verification"],
                                                 "img_num_" + str(cell.img_number) + "__cell_num_" + str(cell.number)
                                                 + "_" + part + "_max_projection_initial.png")
    max_projection_rotated_file_path = os.path.join(folders["rotatation_verification"],
                                                    "img_num_" + str(cell.img_number) + "__cell_num_" + str(cell.number)
                                                    + "_" + part + "_max_projection_rotated.png")
    hough_lines_file_path = os.path.join(folders["rotatation_verification"],
                                         "img_num_" + str(cell.img_number) + "__cell_num_" + str(cell.number)
                                         + "_" + part + "_hough_lines.png")
    cv2.imwrite(max_projection_init_file_path, max_progection_img)
    cv2.imwrite(max_projection_rotated_file_path, rotated_max_projection)
    cv2.imwrite(hough_lines_file_path, hough_lines_img)

    # save cross section for verification
    middle_xsection_file_path = os.path.join(folders["middle_xsection"],
                                             "img_num_" + str(cell.img_number) + "__cell_num_" + str(cell.number)
                                             + "_" + part + "_middle_xsection.png")
    cv2.imwrite(middle_xsection_file_path, mid_cut_img)


def find_biggest_nucleus_layer(temp_folders, treshold, find_biggest_mode, unet_parm=None):
    """
    Finds and analyzes image (layer) with the biggest area of the nucleus
    ---
        Parameters:
        - input_folder (string): path to the folder where all slices of the nucleus
                            in jpg format is located
    ---
        Returns:
        - biggest_nucleus_mask (np. array): array of 0 and 1 where 1 is white pixels
                                        which represent the shape of the biggest area
                                        of nucleus over all layers and 0 is a background
    """
    nucleus_area = 0
    mask, center = None, None

    if find_biggest_mode == "unet":
        run_predict_unet(temp_folders["raw"], temp_folders["nucleus_top_mask"], unet_parm.from_top_nucleus_unet_model,
                         unet_parm.unet_model_scale,
                         unet_parm.unet_model_thrh)
        folder = temp_folders["nucleus_top_mask"]
    elif find_biggest_mode == "trh":
        folder = temp_folders["raw"]

    for img_path in glob.glob(os.path.join(folder, "*_nucleus_*.png")):
        nucleus_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        current_nucleus_cnt = Contour.get_biggest_cnt(nucleus_img, treshold)
        if current_nucleus_cnt is None:
            continue
        current_nucleus_cnt_area = cv2.contourArea(current_nucleus_cnt)
        if current_nucleus_cnt_area > nucleus_area:
            nucleus_area = current_nucleus_cnt_area
            mask = Contour.draw_cnt(current_nucleus_cnt, nucleus_img.shape[:2])
    return mask


def сut_out_mask(mask, input_folder, output_folder, identifier):
    """
    Cuts out an area that corresponds to the mask on each image (layer) located in the input_folder,
    saves processed images in the output_folder, and returns processed images combined into image_3d
    ---
        Parameters:
        - input_folder (string): path to the input folder with jpg images
        - output_folder (string): path to the folder to save processed jpg images
        - identifier (string): "actin" or "nucleus"
        - mask (np. array): stencil to cut out from the images
    ---
        Returns:
        - image_3d (np. array): three-dimensional array of processed (cut out) images combined layer by layer together

    """
    object_layers = []
    for img_path in glob.glob(os.path.join(input_folder, "*_" + identifier + "_*.png")):
        layer = int(img_path.rsplit(".", 1)[0].rsplit("_", 1)[1])
        object_img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        mask[mask == 255] = 1
        object_layer = np.multiply(object_img, mask)
        object_layers.append([object_layer, layer])
        cv2.imwrite(os.path.join(output_folder, os.path.basename(img_path)), object_layer)

    object_layers = sorted(object_layers, key=lambda x: x[1], reverse=True)
    image_3d = np.asarray([img for img, layer in object_layers], dtype=np.uint8)
    image_3d = np.moveaxis(image_3d, 0, -1)
    return image_3d


def plot_histogram(title, image):
    histogram, bin_edges = np.histogram(image, bins=256 * 256)
    plt.figure()
    plt.title(title)
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")

    plt.plot(histogram)  # <- or here
    plt.show()


def is_point_in_pyramid_old_version(x, y, z, x_test, y_test, z_test, con_angle, min_len, resolution):
    is_x_in_pyramid = x_test < x + min_len
    # Find y frame based on known angle and the height of the triangle (con_angle)
    sin = math.sin(math.radians(con_angle))
    y_frame = abs(sin * (x - x_test))
    is_y_in_pyramid = y - y_frame <= y_test <= y + y_frame
    z_frame = int(y_frame * (
                resolution.y / resolution.z))  # Since pixel size is different for xy and xz planes, adjustment should be done
    is_z_in_pyramid = z - z_frame <= z_test <= z + z_frame
    return is_x_in_pyramid and is_y_in_pyramid and is_z_in_pyramid


def is_point_in_pyramid(right_rotated_xy, right_rotated_xz, left_rotated_xy, left_rotated_xz, con_angle, con_angle_z,
                        min_len, resolution):
    x_y, y = right_rotated_xy
    x_y_test, y_test = left_rotated_xy
    x_z, z = right_rotated_xz
    x_z_test, z_test = left_rotated_xz
    is_x_y_in_pyramid = x_y_test < x_y + min_len
    # Find y frame based on known angle and the height of the triangle (con_angle)
    sin = math.sin(math.radians(con_angle))
    sin_z = math.sin(math.radians(con_angle_z))
    y_frame = abs(sin * (x_y - x_y_test))
    is_y_in_pyramid = y - y_frame <= y_test <= y + y_frame
    z_frame = int(abs(sin_z * (x_z - x_z_test)) * (
                resolution.y / resolution.z))  # Since pixel size is different for xy and xz planes, adjustment should be done
    is_x_z_in_pyramid = x_z_test < x_z + min_len
    is_z_in_pyramid = z - z_frame <= z_test <= z + z_frame
    return is_x_y_in_pyramid and is_y_in_pyramid and is_x_z_in_pyramid and is_z_in_pyramid


def rotate_point(rotated_point, main_point, rot_angle):
    """
    Based on https://math.stackexchange.com/questions/3244392/getting-transformed-rotated-coordinates-relative-to
    -the-center-of-a-rotated-ima example the new coordinates of the candidates' starting point (p1,p2) after a rotation by θ
    degrees around the ending point of the main line (a,b): (a + x cos θ − y sin θ,   b + x sin θ + y cos θ)
    where x = p1−a, y = p2−b
    """
    x = rotated_point[0] - main_point[0]
    y = rotated_point[1] - main_point[1]
    new_coordinates = (
    int(main_point[0] + x * math.cos(math.radians(rot_angle)) - y * math.sin(math.radians(rot_angle))),
    int(main_point[1] + x * math.sin(math.radians(rot_angle)) + y * math.cos(math.radians(rot_angle))))
    return new_coordinates


def normalization(img, norm_th):
    img[np.where(img > norm_th)] = norm_th
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return img


def get_nuclei_masks(temp_folders, analysis_folder, image_path, nuc_theshold,
                     nuc_area_min_pixels_num, find_biggest_mode, img_num, unet_parm=None):
    img_base_path = os.path.splitext(os.path.basename(image_path))[0]
    max_projection_origin_size, max_progection_unet_size = find_max_projection(temp_folders["raw"], "nucleus",
                                                                               show_img=False)
    max_projection_path = os.path.join(temp_folders["nucleus_top_img"], img_base_path + ".png")
    cv2.imwrite(max_projection_path, max_projection_origin_size)
    dim = 0
    cnts = []
    if find_biggest_mode == "unet":
        run_predict_unet(temp_folders["nucleus_top_img"], temp_folders["nucleus_top_mask"],
                         unet_parm.from_top_nucleus_unet_model,
                         unet_parm.unet_model_scale,
                         unet_parm.unet_model_thrh)
        mask_path = os.path.join(temp_folders["nucleus_top_mask"], img_base_path + ".png")
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.dilate(mask_img, kernel, iterations=1)
        cnts = Contour.get_img_cnts(img_dilation, nuc_theshold)
        dim = img_dilation.shape

    elif find_biggest_mode == "trh":
        img_path = os.path.join(temp_folders["nucleus_top_img"], img_base_path + ".png")
        nucleus_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        cnts = Contour.get_img_cnts(nucleus_img, nuc_theshold)
        dim = nucleus_img.shape

    # Remove nuclei that touch edges of the image
    cnts = remove_edge_nuc(cnts, dim)
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > nuc_area_min_pixels_num]  # removes noise

    nuclei_masks = []

    for i, cnt in enumerate(cnts):
        one_nuc_mask = np.zeros(dim, dtype="uint8")
        cv2.drawContours(one_nuc_mask, [cnt], -1, color=(255, 255, 255), thickness=cv2.FILLED)
        one_nuc_mask_path = os.path.join(temp_folders["nuclei_top_masks"], img_base_path + "_" + str(i) + ".png")
        cv2.imwrite(one_nuc_mask_path, one_nuc_mask)
        nuclei_masks.append(one_nuc_mask)

    draw_and_save_cnts_verification(analysis_folder, image_path, cnts, max_projection_origin_size, img_num)

    return nuclei_masks


def draw_and_save_cnts_verification(analysis_folder, image_path, cnts, max_progection_img, img_num):
    base_img_name = os.path.splitext(os.path.basename(image_path))[0]
    ver_img_path = os.path.join(analysis_folder["area_ver"],
                                base_img_name + "_img-num_" + str(img_num) + "_max_projection.png")

    cv2.drawContours(max_progection_img, cnts, -1, (255, 255, 255), 5)

    for i, cnt in enumerate(cnts):
        org = Contour.get_cnt_center(cnt)
        cv2.putText(max_progection_img, str(i), org, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 255, 0),
                    thickness=3)
    cv2.imwrite(ver_img_path, max_progection_img)


def remove_edge_nuc(cnts, img_dim):  # removes cells that touch the edges of the frame
    max_x, max_y = img_dim

    # Countours should not touch the edges of the image
    new_cnts = []
    for cnt in cnts:
        cnt_max_x = cnt[:, 0, 0].max()
        cnt_max_y = cnt[:, 0, 1].max()
        cnt_min_x = cnt[:, 0, 0].min()
        cnt_min_y = cnt[:, 0, 1].min()
        if (cnt_max_x < max_x - 2) and (cnt_max_y < max_y - 2) and (cnt_min_x > 1) and (cnt_min_y > 1):
            new_cnts.append(cnt)

    return new_cnts
