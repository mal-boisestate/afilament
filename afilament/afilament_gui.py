import tkinter as tk
from tkinter.messagebox import showinfo
from tkinter import filedialog
import customtkinter as ctk
import javabridge
import bioformats
import time
import pickle
import logging
import json
import os
from types import SimpleNamespace

from afilament.objects.CellAnalyser import CellAnalyser

def run_through_gui(bioformat_imgs_path, output_folder,
                            nuc_channel, actin_channel,
                            fiber_min_layers_theshold, isSave_obj):
    # Load JSON configuration file. This file can be produced by GUI in the future implementation
    with open("config.json", "r") as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    # Add user input into configuration file:
    config.actin_channel = actin_channel
    config.nucleus_channel = nuc_channel
    config.fiber_min_layers_theshold = fiber_min_layers_theshold
    config.confocal_img = bioformat_imgs_path
    config.output_analysis_path = output_folder
    # Specify image numbers to be analyzed
    img_nums = range(1, 2)


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
        if isSave_obj:
            with open('analysis_data/cells_objects.pickle', "wb") as file_to_save:
                pickle.dump(all_cells, file_to_save)

    # Save individual cell data to CSV file
    analyser.save_cells_data(all_cells)
    # Save aggregated cell statistics to CSV file
    analyser.save_aggregated_cells_stat(all_cells)
    # Save current configuration settings to JSON file
    analyser.save_config()

    end = time.time()
    print("Total time is: ")
    print(end - start)

    #Move analysis results to the folder specified by user:
    #1. Move actin_objects
    os.rename('old_directory/test_file.txt', 'new_directory/test_file.txt')

    # Kill Java virtual machine
    javabridge.kill_vm()



def show_data_page():
    data_frame.pack(expand=True)
    window_width = 650
    window_height = 600

    # get the screen dimension
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # find the center point
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)

    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')


def run_analysis():

    print(f"Input folder: {data_page.input_folder.get()}")
    bioformat_imgs_path = data_page.input_folder.get()

    print(f"Output folder: {data_page.output_folder.get()}")
    output_folder = data_page.output_folder.get()

    print(f"Mask channel: {data_page.nuc_channel.get()}")
    nuc_channel = data_page.nuc_channel.get()

    print(f"Mask channel: {data_page.actin_channel.get()}")
    actin_channel = data_page.actin_channel.get()

    print(f"Minimal nucleus area: {data_page.min_fiber_len.get()}")
    fiber_min_layers_theshold = data_page.min_fiber_len.get()

    print(f"Save object: {data_page.save_obj.get()}")
    isSave_obj = True if data_page.save_obj.get() == "true" else False



    if bioformat_imgs_path == "" or output_folder == "" or fiber_min_layers_theshold == "":
        showinfo(
            title='Information',
            message="Please fill all blanks to make the program run"
        )

    else:
        showinfo(
            title='Information',
            message=f"The analysis process is about to begin. Once completed, you can find the results "
                    f"in the designated output folder: {output_folder}. Please note that the analysis "
                    f"can take a significant amount of time, depending on the size of the dataset and "
                    f"the capabilities of your machine."

            )
        try:
            root.destroy()
            run_through_gui(bioformat_imgs_path, output_folder,
                            int(nuc_channel), int(actin_channel),
                            int(fiber_min_layers_theshold), isSave_obj)
        except Exception as e:
            a = 1


class DataCollectionPage:
    def __init__(self, data_root):
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.nuc_channel = tk.StringVar()
        self.actin_channel = tk.StringVar()
        self.min_fiber_len = tk.DoubleVar()
        self.save_obj = tk.StringVar()

        # configure the grid for data root label
        data_root.columnconfigure(0, weight=1)
        data_root.columnconfigure(1, weight=2)

        # input folder path
        ctk.CTkLabel(master=data_root, text='Images path:', anchor='w').grid(column=0, row=1, sticky=tk.W, padx=15, pady=15)
        input_parent = ctk.CTkLabel(master=data_root)
        input_parent.grid(column=1, row=1, sticky=tk.EW)
        input_parent.columnconfigure(0, weight=10)
        input_parent.columnconfigure(1, weight=1)

        input_textbox = ctk.CTkEntry(master=input_parent, textvariable=self.input_folder)
        input_textbox.grid(row=0, column=0, sticky=tk.EW)
        button_br_1 = ctk.CTkButton(master=input_parent, text="...", command=self.input_button, width=30)
        button_br_1.grid(row=0, column=1, padx=5)

        # output folder path
        ctk.CTkLabel(master=data_root, text='Save results at:', anchor='w').grid(column=0, row=2, sticky=tk.W, padx=15, pady=15)
        input_parent = ctk.CTkLabel(master=data_root)
        input_parent.grid(column=1, row=2, sticky=tk.EW)
        input_parent.columnconfigure(0, weight=10)
        input_parent.columnconfigure(1, weight=1)

        input_textbox = ctk.CTkEntry(master=input_parent, textvariable=self.output_folder)
        input_textbox.grid(row=0, column=0, sticky=tk.EW)
        button_br_2 = ctk.CTkButton(master=input_parent, text="...", command=self.output_button, width=30)
        button_br_2.grid(row=0, column=1, padx=5)

        # Nucleus channel number
        ctk.CTkLabel(master=data_root, text='Nucleus channel', anchor='w').grid(column=0, row=4, sticky=tk.W, padx=15,
                                                                                pady=15)
        mask_combobox_nuc = ctk.CTkComboBox(master=data_root, values=["0", "1", "2", "3"], variable=self.nuc_channel)
        mask_combobox_nuc['values'] = ["0", "1", "2", "3"]
        
        # prevent typing a value
        mask_combobox_nuc['state'] = 'readonly'
        mask_combobox_nuc.grid(column=1, row=4, sticky=tk.W, padx=0, pady=15)
        mask_combobox_nuc.get()

        # Actin channel number
        ctk.CTkLabel(master=data_root, text='Actin channel', anchor='w').grid(column=0, row=5, sticky=tk.W, padx=15,
                                                                                pady=15)
        mask_combobox_actin = ctk.CTkComboBox(master=data_root, values=["0", "1", "2", "3"], variable=self.actin_channel)
        mask_combobox_actin['values'] = ["0", "1", "2", "3"]

        # prevent typing a value
        mask_combobox_actin['state'] = 'readonly'
        mask_combobox_actin.grid(column=1, row=5, sticky=tk.W, padx=0, pady=15)
        mask_combobox_actin.get()

        #minimal nucleus area
        ctk.CTkLabel(master=data_root, text='Minimal fiber length (pixels)', anchor='w').grid(column=0, row=7, sticky=tk.W, padx=15, pady=15)
        spin_box = ctk.CTkEntry(master=data_root, width=50, textvariable=self.min_fiber_len)
        spin_box.grid(column=1, row=7, sticky=tk.W, padx=0, pady=15)

        #Save nucleus object

        separate_check = ctk.CTkCheckBox(master=data_root,
                                         text='Save object file',
                                         variable=self.save_obj,
                                         onvalue='true',
                                         offvalue='false')
        separate_check.grid(column=0, row=8, sticky=tk.W, padx=30, pady=15, columnspan=2)



        self.analize_button = ctk.CTkButton(
            master=data_root,
            text="Analyze")
        self.analize_button.grid(column=1, row=9, sticky=tk.E, padx=15, pady=30)


    def input_button(self):
        self.input_folder.set(filedialog.askdirectory())

    def output_button(self):
        self.output_folder.set(filedialog.askdirectory())

# design customization
ctk.set_appearance_mode("System")  # Modes: system (default), light, dark
ctk.set_default_color_theme("green")  # Themes: blue (default), dark-blue, green

root = ctk.CTk()
root.title('Afilament')
window_width = 370
window_height = 430

# get the screen dimension
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# find the center point
center_x = int(screen_width/2 - window_width / 2)
center_y = int(screen_height/2 - window_height / 2)

root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
root.iconbitmap(r'..\docs\imgs\favicon.ico')


# Data collection
data_frame = ctk.CTkFrame(master=root)
data_page = DataCollectionPage(data_frame)
data_page.analize_button.configure(command=run_analysis)
data_frame.pack(expand=True)

root.mainloop()


