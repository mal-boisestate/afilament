import time
import pickle
from afilament.objects.CellAnalyser import CellAnalyser
from afilament.objects import Utils
from afilament.objects.Parameters import UnetParam, ImgResolution
from pathlib import Path
import javabridge
import bioformats
import logging


def main():
    RECALCULATE = True
    img_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # 2 nuc thershold 30 does not work
    nucleus_channel = 0  # 1 for original czi file
    actin_channel = 1 # 0 for original czi file
    # confocal_img = r"D:\BioLab\cells\Confocal_img\2022.05.25_leica_DAPI_488\2022.05.25_MSC_Control_Series-01-20.lif"
    confocal_img = r"F:\Nina_Docs\Img\Confocal\2022.10.19_MSC_6xLIV_10_img_32_cells.lif" # Path to folder(czi) or file (lif)
    nuc_theshold = 60
    fiber_min_layers_theshold = 10 #in pixels
    node_actin_len_th = 2 #for node creation, do not breake actin if one of the part is too small

    actin_unet_model = r"../unet/models/CP_epoch200_actin_weight.corection_200_labling_V2.pth"
    nucleus_unet_model = r"../unet/models/CP_epoch50_nucleus_weight.corection_200_labling_WITH_FLAT_Apatome_no_agum_V3.pth"
    from_top_nucleus_unet_model = r"../unet/models/CP_epoch200_max_pr.pth"
    unet_model_scale = 1
    unet_img_size = (512, 512)
    unet_model_thrh = 0.5
    unet_parm = UnetParam(from_top_nucleus_unet_model, nucleus_unet_model, actin_unet_model, unet_model_scale, unet_model_thrh, unet_img_size)
    is_plot_fibers = False
    is_plot_nodes = False
    is_separate_cap_bottom = True
    is_auto_normalized = False
    is_connect_fibers = False
    norm_th = 2**16 #when auto chose it will be recalculated format is tuple example (1000, 1000)
    find_biggest_mode = "trh" #"unet" or "trh"
    fiber_joint_angle = 10
    fiber_joint_distance = 50
    nuc_area_min_pixels_num = 1000
    javabridge.start_vm(class_path=bioformats.JARS)

    analyser = CellAnalyser(nucleus_channel, actin_channel, confocal_img,
                            nuc_theshold, unet_parm, fiber_min_layers_theshold, node_actin_len_th,
                            is_plot_fibers, is_plot_nodes, is_auto_normalized,
                            norm_th, find_biggest_mode,
                            is_separate_cap_bottom, is_connect_fibers,
                            fiber_joint_angle, fiber_joint_distance, nuc_area_min_pixels_num)

    start = time.time()
    all_cells = []

    logging.basicConfig(filename='myapp.log', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)

    if RECALCULATE:
        for img_num in img_nums:
            try:
                        cells = analyser.analyze_img(img_num)
                        all_cells.extend(cells)
            except Exception as e:
                logger.error(f"\n----------- \n Img #{img_num} from file {confocal_img} was not analysed. "
                                     f"\n Error: {e} \n----------- \n")
                print("An exception occurred")

        with open('analysis_data/test_cells_bach.pickle', "wb") as file_to_save:
            pickle.dump(all_cells, file_to_save)
    else:
        all_cells = pickle.load(open('analysis_data/test_cells_bach.pickle', "rb"))

    # analyser.save_cells_data(cells)
    # analyser.save_aggregated_cells_stat(cells)

    analyser.save_cells_data(all_cells)
    analyser.save_aggregated_cells_stat(all_cells)

    end = time.time()
    print("Total time is: ")
    print(end - start)
    javabridge.kill_vm()


if __name__ == '__main__':
    main()