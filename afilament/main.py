import time
import pickle
import javabridge
import bioformats
import logging
import json
from types import SimpleNamespace

from afilament.objects.CellAnalyser import CellAnalyser

def main():
    # Load JSON configuration file. This file can be produced by GUI in the future implementation
    with open("config.json", "r") as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    # Specify image numbers to be analyzed
    img_nums = range(1, 2)

    # Set RECALCULATE to True to re-run analysis on all images
    # Set RECALCULATE to False to load previously analyzed data
    RECALCULATE = True

    # Start Java virtual machine for Bioformats library
    javabridge.start_vm(class_path=bioformats.JARS)

    # Initialize CellAnalyser object with configuration settings
    analyser = CellAnalyser(config)

    start = time.time()
    all_cells = []

    # Set up logging to record errors
    logging.basicConfig(filename='myapp.log', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)

    if RECALCULATE:
        # Analyze each specified image and store cell data in all_cells list
        for img_num in img_nums:
            try:
                cells = analyser.analyze_img(img_num)
                all_cells.extend(cells)
            except Exception as e:
                # Log error message if analysis fails for an image
                logger.error(f"\n----------- \n Img #{img_num} from file {config.confocal_img} was not analysed. "
                                     f"\n Error: {e} \n----------- \n")
                print("An exception occurred")

        # Save analyzed cell data to a pickle file
        with open('analysis_data/test_cells_bach.pickle', "wb") as file_to_save:
            pickle.dump(all_cells, file_to_save)
    else:
        # Load previously analyzed cell data from a pickle file
        all_cells = pickle.load(open('analysis_data/test_cells_bach.pickle', "rb"))

    # Save individual cell data to CSV file
    analyser.save_cells_data(all_cells)
    # Save aggregated cell statistics to CSV file
    analyser.save_aggregated_cells_stat(all_cells)
    # Save current configuration settings to JSON file
    analyser.save_config()

    end = time.time()
    print("Total time is: ")
    print(end - start)

    # Kill Java virtual machine
    javabridge.kill_vm()

if __name__ == '__main__':
    main()