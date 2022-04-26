import os
import cv2.cv2 as cv2
import numpy as np

from afilament.objects import Utils
from unet.predict import run_predict_unet
from afilament.objects import Contour


class Nucleus(object):
    """
    Creates an object that track all nucleus information
    """

    def __init__(self):
        self.nuc_volume = None
        self.nuc_length = None
        self.nuc_width = None
        self.nuc_high = None
        self.nuc_3D = None
        self.point_cloud = None

    def reconstruct(self, rot_angle, cnt_extremes, folders, unet_parm, resolution):
        """
        Reconstruct nucleus and saves all stat info into this Nicleus objects
        ---
        Parameters:
        - rot_angle (int): rotation angle to rotate images of the nucleus before making cross-section images for segmentation
        - rotated_cnt_extremes (CntExtremes object):
        - folders: (list): list of folders to save intermediate images during analysis
        - unet_parm (UnetParam object):
        img_resolution (ImgResolution object):
        """
        print("\nFinding nuclei...")
        if len(os.listdir(folders["cut_out_nuc"])) == 0:
            raise ValueError(f"Can not reconstruct nucleus, the directory {folders['cut_out_nuc']} that should "
                             f"include preprocessed images is empty")
        nucleus_3d_img, _ = Utils.rotate_and_get_3D(folders["cut_out_nuc"], 'nucleus', rot_angle)
        Utils.get_yz_xsection(nucleus_3d_img, folders["nucleous_xsection"], 'nucleus', cnt_extremes)
        run_predict_unet(folders["nucleous_xsection"], folders["nucleus_mask"], unet_parm.nucleus_unet_model,
                         unet_parm.unet_model_scale,
                         unet_parm.unet_model_thrh)
        self.nuc_3D = Utils.get_3d_img(folders["nucleus_mask"])
        volume, self.point_cloud = self.nucleus_reco_3d(resolution)
        length = (cnt_extremes.right[0] - cnt_extremes.left[0]) * resolution.x
        width = (cnt_extremes.bottom[1] - cnt_extremes.top[1]) * resolution.y
        self.nuc_volume = volume
        self.nuc_length = length
        self.nuc_width = width
        self.nuc_high = None

    def nucleus_reco_3d(self, resolution):
        points = []
        center_x, center_y, center_z = self.get_nucleus_origin()
        print(center_x, center_y, center_z)

        xdata, ydata, zdata = [], [], []
        volume = 0
        for slice in range(self.nuc_3D.shape[0]):
            xsection_img = self.nuc_3D[slice, :, :]

            slice_cnts = cv2.findContours(xsection_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            if len(slice_cnts) != 0:
                slice_cnt = slice_cnts[np.argmax([len(cnt) for cnt in slice_cnts])]
                volume += cv2.contourArea(
                    slice_cnt) * resolution.x * resolution.y * resolution.z  # calculate volume based on counter area
                # volume += np.count_nonzero(utils.draw_cnts((512, 512), [slice_cnt])) * scale_x * scale_y * scale_z #calculate volume based on pixel number

                if slice % 15 == 0:
                    ys = [pt[0, 0] for idx, pt in enumerate(slice_cnt) if idx % 4 == 0 and pt[0, 0] < 720]  # 720
                    zs = [pt[0, 1] for idx, pt in enumerate(slice_cnt) if idx % 4 == 0 and pt[0, 0] < 720]

                    xdata.extend([slice] * len(ys))
                    ydata.extend(ys)
                    zdata.extend(zs)

                cnt_ys = slice_cnt[:, 0, 1]
                cnt_zs = slice_cnt[:, 0, 0]

                points.extend([[x - center_x, y - center_y, center_z - z] for x, y, z in
                               zip([slice] * len(cnt_ys), cnt_ys, cnt_zs)])

        np.savetxt("nucleus_points_coordinates.csv", np.array(points, dtype=int), delimiter=",", fmt="%10.0f")

        print("Nucleus volume: {}".format(volume))

        # TODO: to print out nucleus data commented this out
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(ydata, zdata, xdata, cmap='Greens', alpha=0.5)
        # plt.show()
        return volume, points

    def get_nucleus_origin(self):
        """
        Finds coordinate of nucleus anchor position which is:
        x is x of the center of the nucleus
        y is y of the center of the nucleus
        z is z the bottom of the nucleus
        ---
            Returns:
            - center_x, center_y, center_z (int, int, int) coordinates of the nucleus anchor position
        """
        center_x = self.nuc_3D.shape[0] // 2
        slice_cnts = cv2.findContours(self.nuc_3D[center_x, :, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        slice_cnt = slice_cnts[np.argmax([len(cnt) for cnt in slice_cnts])]
        cnt_extremes = Contour.get_cnt_extremes(slice_cnt)
        center_y = self.nuc_3D.shape[1] // 2
        center_z = cnt_extremes.right[0]

        return center_x, center_y, center_z
