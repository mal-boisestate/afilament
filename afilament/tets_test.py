import tkinter as tk
import customtkinter as ctk


def show_frame(type):
    if type == "new":
        print("New analysis")
        frame2.tkraise()
    elif type == "recalculate":
        frame3.tkraise()


# design customization
ctk.set_appearance_mode("System")  # Modes: system (default), light, dark
ctk.set_default_color_theme("green")  # Themes: blue (default), dark-blue, green


window = ctk.CTk()
window.state('zoomed')

window.rowconfigure(0, weight=1)
window.columnconfigure(0, weight=1)

frame1 = tk.Frame(window)
frame2 = tk.Frame(window)
frame3 = tk.Frame(window)

for frame in (frame1, frame2, frame3):
    frame.grid(row=0, column=0, sticky='nsew')

# ==================Frame 1 code
frame1_title = tk.Label(frame1, text='Page 1', font='times 35', bg='red')
frame1_title.pack(fill='both', expand=True)

selected_type = tk.StringVar()
analysis_types = (('Run new analysis', 'new'),
                    ('Recalculate statistics', 'recalculate'))

# label
label = ctk.CTkLabel(master=frame1, text="Choose Type of Analysis: ")
label.pack(fill='x', padx=10, pady=15)

# Analysis type radio buttons
for type in analysis_types:
    r = ctk.CTkRadioButton(
        frame1,
        text=type[0],
        value=type[1],
        variable=selected_type,
        # command=lambda: show_frame(selected_type.get())
    )
    r.pack(fill='x', padx=50, pady=5)

#
frame1_btn = tk.Button(frame1, text='Enter', command=lambda: show_frame(selected_type.get()))
frame1_btn.pack(fill='x', ipady=15)

# ==================Frame 2 code
frame2_title = tk.Label(frame2, text='Page 2', font='times 35', bg='yellow')
frame2_title.pack(fill='both', expand=True)

# ==================Frame 3 code
frame3_title = tk.Label(frame3, text='Page 3', font='times 35', bg='green')
frame3_title.pack(fill='both', expand=True)


frame1.tkraise()

window.mainloop()