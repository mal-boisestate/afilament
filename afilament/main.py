import time
import pickle
import javabridge
import bioformats
import logging
import json
import os
import sys
import argparse
import glob
from types import SimpleNamespace
from pathlib import Path

from objects.CellAnalyser import CellAnalyser
from objects.Parameters import ImgResolution, CellsImg
from objects import Utils

sys.path.insert(0, 'D:\BioLab\scr_2.0')


class JavaVM:
    def __enter__(self):
        # Set the directory for JVM crash logs
        crash_log_directory = os.path.abspath("err_logs")
        os.environ['JAVA_TOOL_OPTIONS'] = f'-XX:ErrorFile={crash_log_directory}/hs_err_pid%p.log'

        javabridge.start_vm(class_path=bioformats.JARS)
        javabridge.static_call("loci/common/DebugTools", "setRootLevel", "(Ljava/lang/String;)V", "OFF")
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        javabridge.kill_vm()
def parse_arguments():
    parser = argparse.ArgumentParser(description="Cell Analysis Script")
    parser.add_argument('img_path', type=str, nargs='?', default=None, help='Path to the confocal image file')
    args = parser.parse_args()
    return args.img_path

def load_config(config_path, img_path=None):
    with open(config_path, "r") as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    if img_path:
        config.confocal_img = img_path
    return config


def setup_logging():
    log_directory = "err_logs"
    log_file_name = "myapp.log"

    # Create 'err_logs' directory if it doesn't exist
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Set up logging with the correct file path
    log_file_path = os.path.join(log_directory, log_file_name)
    logging.basicConfig(filename=log_file_path, level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    return logging.getLogger(__name__)

def main():

    img_path = parse_arguments()
    config = load_config("config.json", img_path)
    logger = setup_logging()
    fiber_min_for_recalc = config.fiber_min_threshold_microns
    RECALCULATE = True

    # Initialize CellAnalyser object with configuration settings
    # There are some outputs for example "nuc area verification" or "cell_img_objects", that should be saved when we
    start = time.time()

    if RECALCULATE:
        # RECALCULATE all data, other statistics is different based on current fiber_length specification, so these statstics
        # can be saved in different folder. So I need to create convinient folder structure

        with JavaVM():

            analyser = CellAnalyser(config)

            Utils.prepare_folder(config.imgs_objects)
            # Analyze each specified image and store cell data in all_cells list
            for img_num, czi_file in enumerate(glob.glob(os.path.join(config.confocal_img, '*.czi'))):
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


if __name__ == '__main__':
    main()