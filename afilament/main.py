import time
import pickle

from afilament.objects.CellAnalyser import CellAnalyser
from afilament.objects import Utils
from afilament.objects.Parameters import UnetParam, ImgResolution




def main():
    RECALCULATE = True
    cell_nums = [2]
    # 2 nuc thershold 30 does not work
    nucleus_channel = 0  # 1 for original czi file
    actin_channel = 1  # 0 for original czi file
    confocal_img_folder = r"D:\BioLab\img\2022.05.10_DAPI_488"
    nuc_theshold = 30
    fiber_min_layers_theshold = 10 #in pixels
    node_actin_len_th = 2 #for node creation, do not breake actin if one of the part is too small

    actin_unet_model = r"D:\BioLab\models\actin\CP_epoch200_actin_weight.corection_200_labling_V2.pth"
    nucleus_unet_model = r"D:\BioLab\models\one_nucleus\CP_epoch50_nucleus_weight.corection_200_labling_WITH_FLAT_Apatome_no_agum_V3.pth"
    from_top_nucleus_unet_model = r"D:\BioLab\models\one_nucleus_top_view\CP_epoch200_nucleus_weight.corection_2_labling_512_512_with_agum.pth"
    unet_model_scale = 1
    unet_img_size = (512, 512)
    unet_model_thrh = 0.5
    unet_parm = UnetParam(from_top_nucleus_unet_model, nucleus_unet_model, actin_unet_model, unet_model_scale, unet_model_thrh, unet_img_size)
    is_plot_fibers = True
    is_plot_nodes = True
    is_auto_normalized = False
    norm_trh = None #when auto chose it will be recalculated
    find_biggest_mode = "unet" #"unet" or "trh"


    scale_x = 0.103  # 0.05882
    scale_y = 0.103  # 0.05882
    scale_z = 0.330  # 0.270
    img_resolution = ImgResolution(scale_x, scale_y, scale_z)
    analyser = CellAnalyser(nucleus_channel, actin_channel, confocal_img_folder, nuc_theshold, unet_parm,
                            img_resolution, fiber_min_layers_theshold, node_actin_len_th,
                            is_plot_fibers, is_plot_nodes, is_auto_normalized, cell_nums, norm_trh, find_biggest_mode)

    start = time.time()
    cells = []
    if RECALCULATE:
        for cell_num in cell_nums:
            cell = analyser.analyze_cell(cell_num, cap=True, bottom=True)
            cells.append(cell)
        with open('analysis_data/test_cells_bach.pickle', "wb") as file_to_save:
            pickle.dump(cells, file_to_save)
    else:
        cells = pickle.load(open('analysis_data/test_cells_bach.pickle', "rb"))

    analyser.save_cells_data(cells)
    analyser.save_aggregated_cells_stat(cells)
    end = time.time()
    print("Total time is: ")
    print(end - start)

if __name__ == '__main__':
    main()