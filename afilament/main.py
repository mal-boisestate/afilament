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

    img_nums = range(1,21)
    RECALCULATE = True

    javabridge.start_vm(class_path=bioformats.JARS)

    analyser = CellAnalyser(config)

    start = time.time()
    all_cells = []

    logging.basicConfig(filename='myapp.log', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)

    if RECALCULATE:
        for img_num in img_nums:
            cells = analyser.analyze_img(img_num)
            all_cells.extend(cells)
            # try:
            #     cells = analyser.analyze_img(img_num)
            #     all_cells.extend(cells)
            # except Exception as e:
            #     logger.error(f"\n----------- \n Img #{img_num} from file {config.confocal_img} was not analysed. "
            #                          f"\n Error: {e} \n----------- \n")
            #     print("An exception occurred")

        # with open('analysis_data/test_cells_bach.pickle', "wb") as file_to_save:
        #     pickle.dump(all_cells, file_to_save)
    else:
        all_cells = pickle.load(open('analysis_data/test_cells_bach.pickle', "rb"))

    analyser.save_cells_data(all_cells)
    analyser.save_aggregated_cells_stat(all_cells)
    analyser.save_config()

    end = time.time()
    print("Total time is: ")
    print(end - start)
    javabridge.kill_vm()


if __name__ == '__main__':
    main()