import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pathlib
import logging
from feature_matching import start_cross_match, start_exhaustive_match
import threading

class TextHandler(logging.Handler):
    # This class allows you to log to a Tkinter Text or ScrolledText widget
    # Adapted from Moshe Kaplan: https://gist.github.com/moshekaplan/c425f861de7bbf28ef06

    def __init__(self, text):
        # run the regular Handler __init__
        logging.Handler.__init__(self)
        # Store a reference to the Text it will log to
        self.text = text

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text.configure(state='normal')
            self.text.insert(tk.END, msg + '\n')
            self.text.configure(state='disabled')
            # Autoscroll to the bottom
            self.text.yview(tk.END)
        # This is necessary because we can't modify the Text from other threads
        self.text.after(0, append)

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.operation_label = tk.Label(self)
        self.operation_label["text"] = "Select an operation:"
        self.operation_label.pack(side="top")

        self.operation_var = tk.StringVar(value="match")
        self.match_radio = tk.Radiobutton(self, text="Match", variable=self.operation_var, value="match")
        self.cross_match_radio = tk.Radiobutton(self, text="Cross match", variable=self.operation_var, value="cross_match")
        self.match_radio.pack(side="top")
        self.cross_match_radio.pack(side="top")

        self.input_image_folder_label = tk.Label(self)
        self.input_image_folder_label["text"] = "Input image folder:"
        self.input_image_folder_label.pack(side="top")

        self.input_image_folder_entry = tk.Entry(self, width=50)
        self.input_image_folder_button = tk.Button(self, text="Browse", command=self.browse_input_image_folder)
        self.input_image_folder_entry.pack(side="top")
        self.input_image_folder_button.pack(side="top")

        self.panoramic_image_folder_label = tk.Label(self)
        self.panoramic_image_folder_label["text"] = "Panoramic image folder (optional):"
        self.panoramic_image_folder_label.pack(side="top")

        self.panoramic_image_folder_entry = tk.Entry(self, width=50)
        self.panoramic_image_folder_button = tk.Button(self, text="Browse", command=self.browse_panoramic_image_folder)
        self.panoramic_image_folder_entry.pack(side="top")
        self.panoramic_image_folder_button.pack(side="top")

        self.output_dir_label = tk.Label(self)
        self.output_dir_label["text"] = "Output directory:"
        self.output_dir_label.pack(side="top")

        self.output_dir_entry = tk.Entry(self, width=50)
        self.output_dir_button = tk.Button(self, text="Browse", command=self.browse_output_dir)
        self.output_dir_entry.pack(side="top")
        self.output_dir_button.pack(side="top")

        self.matcher_var = tk.StringVar(value="XFeatLighterglue")
        self.matcher_label = tk.Label(self)
        self.matcher_label["text"] = "Matcher:"
        self.matcher_label.pack(side="top")
        self.matcher_xfeatlighterglue_radio = tk.Radiobutton(self, text="XFeatLighterglue", variable=self.matcher_var, value="XFeatLighterglue")
        self.matcher_siftknn_radio = tk.Radiobutton(self, text="SIFTkNN", variable=self.matcher_var, value="SIFTkNN")
        self.matcher_xfeatlighterglue_radio.pack(side="top")
        self.matcher_siftknn_radio.pack(side="top")

        self.recurse_dirs_var = tk.IntVar()
        self.recurse_dirs_checkbox = tk.Checkbutton(self, text="Recurse subdirectories for images", variable=self.recurse_dirs_var)
        self.recurse_dirs_checkbox.pack(side="top")

        self.cache_features_var = tk.IntVar()
        self.cache_features_checkbox = tk.Checkbutton(self, text="Store features in a cache directory (SIFTkNN only)", variable=self.cache_features_var)
        self.cache_features_checkbox.pack(side="top")

        self.debug_var = tk.IntVar()
        self.debug_checkbox = tk.Checkbutton(self, text="Debug", variable=self.debug_var)
        self.debug_checkbox.pack(side="top")

        self.output_text = scrolledtext.ScrolledText(self, width=80, height=10, state='disabled')
        self.output_text.pack(side="top", expand=True, fill="x")

        self.log_handler = TextHandler(self.output_text)

        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger("LiLoc")
        logger.addHandler(self.log_handler)

        self.run_button = tk.Button(self, text="Run", command=self.start_run_thread)
        self.run_button.pack(side="top")

    def browse_input_image_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.input_image_folder_entry.delete(0, tk.END)
            self.input_image_folder_entry.insert(tk.END, path)

    def browse_panoramic_image_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.panoramic_image_folder_entry.delete(0, tk.END)
            self.panoramic_image_folder_entry.insert(tk.END, path)

    def browse_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(tk.END, path)

    def run(self):
        self.lock_input_fields(True)
        operation = self.operation_var.get()
        input_image_folder = pathlib.Path(self.input_image_folder_entry.get())
        panoramic_image_folder = None
        if self.panoramic_image_folder_entry.get():
            panoramic_image_folder = pathlib.Path(self.panoramic_image_folder_entry.get())
        output_dir = pathlib.Path(self.output_dir_entry.get())
        matcher = self.matcher_var.get()
        recurse_dirs = bool(self.recurse_dirs_var.get())
        cache_features = bool(self.cache_features_var.get())
        debug = bool(self.debug_var.get())

        logger = logging.getLogger("LiLoc")
        logger.info("Running %s operation", operation)

        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        args = {
            "output_dir": output_dir,
            "matcher": matcher,
            "recurse_dirs": recurse_dirs,
            "cache_features": cache_features,
            "debug": debug
        }


        if operation == "cross_match":
            if not panoramic_image_folder:
                messagebox.showerror("Error", "Panoramic image folder is required for cross match")
                self.lock_input_fields(False)
                return
            args["panoramic_image_folder"] = panoramic_image_folder
            args["input_image_folder"] = input_image_folder
            start_cross_match(args)
        elif operation == "match":
            args["input_image_folder"] = input_image_folder
            start_exhaustive_match(args)

        logger.info("Operation completed")
        self.lock_input_fields(False)

    def lock_input_fields(self, lock):
        for widget in [
            self.cross_match_radio,
            self.match_radio,
            self.input_image_folder_entry,
            self.input_image_folder_button,
            self.panoramic_image_folder_entry,
            self.panoramic_image_folder_button,
            self.output_dir_entry,
            self.output_dir_button,
            self.matcher_xfeatlighterglue_radio,
            self.matcher_siftknn_radio,
            self.recurse_dirs_checkbox,
            self.cache_features_checkbox,
            self.debug_checkbox,
        ]:
            if lock:
                widget.config(state="disabled")
            else:
                widget.config(state="normal")

    def start_run_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()

root = tk.Tk()
app = Application(master=root)
app.mainloop()