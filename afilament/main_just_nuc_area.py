import time
import javabridge
import bioformats
import logging
import json
from datetime import datetime
from types import SimpleNamespace

from afilament.objects.CellAnalyser import CellAnalyser

def main():
    # Load JSON configuration file. This file can be produced by GUI in the future implementation
    with open("config.json", "r") as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    img_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    javabridge.start_vm(class_path=bioformats.JARS)

    analyser = CellAnalyser(config)

    start = time.time()

    logging.basicConfig(filename='myapp.log', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)

    for img_num in img_nums:
        try:
            analyser.save_nuc_verification(img_num)
        except Exception as e:
            logger.error(f"\n----------- \n Img #{img_num} from file {config.confocal_img} was not analysed. "
                                 f"\n Error: {e} \n----------- \n")
            print("An exception occurred")

    analyser.save_config()
    end = time.time()
    print("Total time is: ")
    print(end - start)
    javabridge.kill_vm()


if __name__ == '__main__':
    main()