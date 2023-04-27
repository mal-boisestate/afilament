import os
import pickle
from pathlib import Path

img_objs_folder = f"img_objects"

for i, file in enumerate(os.listdir(img_objs_folder)):
    # add check if directory is empty and ask user to specify where get data
    img_path = Path(os.path.join(img_objs_folder, file))
    if img_path.suffix == ".pickle":
        print(f"{i}: Success")
    else:
        print(f"{i}: Not pickle")
    a = 1
