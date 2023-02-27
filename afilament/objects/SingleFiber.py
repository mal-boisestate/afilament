import numpy as np
import math

import cv2.cv2 as cv2
from statistics import mean, median



class SingleFiber(object):
    # scale_x = 0.04  # units - micrometer
    # scale_y = 0.04  # units - micrometer (image was resized to account for different scale along z axis)
    # scale_z = 0.17  # units - micrometer (image was resized to account for different scale along z axis)

    def __init__(self, x, y, z, intensity, layer, cnt):
        self.xs = [x]
        self.ys = [y]
        self.zs = [z]
        self.intensities = [intensity]
        self.cnts = [cnt]
        self.last_layer = [layer]
        self.n = 1
        self.line = None
        self.merged = False
        self.part = None

    def update(self, x, y, z, intensity, layer, cnt):
        self.xs.append(x)
        self.ys.append(y)
        self.zs.append(z)
        self.intensities.append(intensity)
        self.cnts.append(cnt)
        self.last_layer.append(layer)
        self.n += 1

    def fit_line(self):
        points = np.asarray([[x, y] for x, y in zip(self.xs, self.ys)])
        self.line = np.squeeze(cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01))

    def assign_cap_or_bottom(self, z_start, z_end, cut_off_coef):
        """
        Check if the fiber is apical (cap) or basal (bottom) based on the mean/median of z coordinates of the given fiber.
        If the mean/median z is located in the upper cut_off_coef part, fiber is apical (cap); otherwise, the fiber is basal(bottom)
        ___
        Parameters:
            - z_start - the start of z coordinates correspond to original images (before padding)
            - the end of z coordinates correspond to original images (before padding)
          ______________________
        |         *            |
        |         * *          |
        |         *  *         |
        |         *   *        |
        |         *   *        |
        |         *  *         |
        |         * *          |
        |         *            |
        ----------------------->
        0   bottom    top    512
                 s     e
                 t     n
                 a     d
                 r
                 t
        """
        mean_z = mean(self.zs)
        median_z = median(self.zs)
        cut_off_coordinate = (z_start + (z_end - z_start) * cut_off_coef)
        print(f"start is {z_start}, end is {z_end}\n")
        if median_z <= cut_off_coordinate:
            self.part = "bottom"
        else:
            self.part = "cap"

    def find_fiber_alignment_angle(self, axis):
        alignment_pix_num = 30
        rot_angle = 0
        if axis == "xy":
            main_line_end_p = (self.xs[-1], self.ys[-1])
            if self.n < alignment_pix_num:
                main_line_start_p = (self.xs[0], self.ys[0])
            else:
                main_line_start_p = (self.xs[-alignment_pix_num], self.ys[-alignment_pix_num])
            angle_sin = (main_line_end_p[1] - main_line_start_p[1]) / np.linalg.norm(np.array(main_line_end_p) - np.array(main_line_start_p))
            rot_angle = - math.degrees(math.asin(angle_sin))
        elif axis == "xz":
            main_line_end_p = (self.xs[-1], self.zs[-1])
            if self.n < alignment_pix_num:
                main_line_start_p = (self.xs[0], self.zs[0])
            else:
                main_line_start_p = (self.xs[-alignment_pix_num], self.zs[-alignment_pix_num])
            angle_sin = (main_line_end_p[1] - main_line_start_p[1]) / np.linalg.norm(np.array(main_line_end_p) - np.array(main_line_start_p))
            rot_angle = - math.degrees(math.asin(angle_sin))

        return rot_angle

    def get_stat(self, resolution):
        """
        actin_length - full length of fiber
        actin_xsection - sum of all xsections for each layer times scale
        actin_volume - actin length times average xsection
        actin_intensity - sum of original intensity (8, 12, or 16 bit) of the actin fiber
        """
        actin_length = (self.xs[-1] - self.xs[0]) * resolution.x
        actin_xsection = np.mean([cv2.contourArea(cnt) for cnt in self.cnts]) * resolution.y * resolution.z
        actin_volume = actin_length * actin_xsection
        actin_intensity = np.sum([intensity for intensity in self.intensities])

        adjacent = (self.xs[-1] - self.xs[0])
        if adjacent == 0:
            adjacent = 1

        return [actin_length, actin_xsection, actin_volume, self.n, actin_intensity]
