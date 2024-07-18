import time
import pickle
import javabridge
import bioformats
import logging
import json
import os
from types import SimpleNamespace
from pathlib import Path

from afilament.objects.CellAnalyser import CellAnalyser
from afilament.objects.Parameters import ImgResolution, CellsImg
from afilament.objects import Utils

def main():

    # Specify image numbers to be analyzed
    img_nums = range(0, 57)

    # Set RECALCULATE to True to re-run analysis on all images
    # Set RECALCULATE to False to load previously analyzed data
    RECALCULATE = True

    # Start Java virtual machine for Bioformats library
    javabridge.start_vm(class_path=bioformats.JARS)

    # Initialize CellAnalyser object with configuration settings
    # There are some outputs for example "nuc area verification" or "cell_img_objects", that should be saved when we
    start = time.time()

    # Load JSON configuration file.
    with open("config.json", "r") as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    fiber_min_for_recalc = config.fiber_min_threshold_microns

    # Set up logging to record errors
    logging.basicConfig(filename='myapp.log', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)


    if RECALCULATE:
        # RECALCULATE all data, other statistics is different based on current fiber_length specification, so these statstics
        # can be saved in different folder. So I need to create convinient folder structure
        analyser = CellAnalyser(config)

        Utils.prepare_folder(config.imgs_objects)
        # Analyze each specified image and store cell data in all_cells list
        for img_num in img_nums:
            try:
                cells, img_name = analyser.analyze_img(img_num)

                # Save analyzed image to a pickle file
                image_data_path = os.path.join(config.imgs_objects, "image_data_" + str(img_num) + ".pickle")
                cells_img = CellsImg(img_name, analyser.img_resolution, cells)
                with open(image_data_path, "wb") as file_to_save:
                    pickle.dump(cells_img, file_to_save)

            except Exception as e:
                # Log error message if analysis fails for an image
                logger.error(f"\n----------- \n Img #{img_num} from file {config.confocal_img} was not analysed. "
                                     f"\n Error: {e} \n----------- \n")
                print("An exception occurred")
    else:
        # If we rerun analysis based on previous data from config file that was in directory
        # we can use only fiber_min_layers_theshold the rest data should be saved
        # as it was used for analysis so we need to load JSON configuration file that was used for specific analysis.
        config_file_path = os.path.join(config.imgs_objects, "analysis_configurations.json")
        with open(config_file_path, "r") as f:
            config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        analyser = CellAnalyser(config)
        analyser.fiber_min_thr_microns = fiber_min_for_recalc


    # Extract statistical data
    # Extract all_cells from images data
    aggregated_stat_list = []



    for file in os.listdir(config.imgs_objects):
        #add check if directory is empty and ask user to specify where get data
        img_path = Path(os.path.join(config.imgs_objects, file))

        #check if it is image file since in this folder we have config file
        if img_path.suffix == ".pickle":
            cells_img = pickle.load(open(img_path, "rb"))

            #Need to be updated based on data saved in img object, since theoreticaly it can be different for each img
            #So we do not save it in config and analyser respectfully
            analyser.img_resolution = cells_img.resolution
            analyser.fiber_min_thr_pixels = analyser.fiber_min_thr_microns / analyser.img_resolution.x


            # Save individual cell data to CSV file
            analyser.save_cells_data(cells_img.cells)
            aggregated_stat_list = analyser.add_aggregated_cells_stat(aggregated_stat_list, cells_img.cells,
                                                                      cells_img.name)

    # Save aggregated cell statistics to CSV file
    analyser.save_aggregated_cells_stat_list(aggregated_stat_list)
    analyser.save_config(RECALCULATE, config.imgs_objects)

    end = time.time()
    print("Total time is: ")
    print(end - start)

    # Kill Java virtual machine
    javabridge.kill_vm()

if __name__ == '__main__':
    main()