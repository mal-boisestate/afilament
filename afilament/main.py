import time
import pickle

from afilament.objects.CellAnalyser import CellAnalyser
from afilament.objects import Utils
from afilament.objects.Parameters import UnetParam, ImgResolution
from pathlib import Path




def main():
    RECALCULATE = True
    cell_nums = [0]
    # 2 nuc thershold 30 does not work
    nucleus_channel = 1  # 1 for original czi file
    actin_channel = 0  # 0 for original czi file
    confocal_img = r"C:\Users\nnina\Desktop\imagies" # Path to folder(czi) or file (lif)
    # confocal_img = r"D:\BioLab\img\Confocal_img\3D_new_set\2022.01.12_KASH_dox.lif" # Path to folder(czi) or file (lif)
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
    norm_th = 5000 #when auto chose it will be recalculated format is tuple example (1000, 1000)
    find_biggest_mode = "trh" #"unet" or "trh"


    analyser = CellAnalyser(nucleus_channel, actin_channel, confocal_img, nuc_theshold, unet_parm,
                            fiber_min_layers_theshold, node_actin_len_th, is_plot_fibers, is_plot_nodes,
                            is_auto_normalized, cell_nums, norm_th, find_biggest_mode)

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