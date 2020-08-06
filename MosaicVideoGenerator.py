import os
import sys
import cv2
import pygame
import subprocess
import numpy as np
from Mosaic import MosaicGenerator

from tkinter import Tk, Frame, Button, Label, Entry
from tkinter import filedialog, colorchooser
from scipy.spatial import Voronoi
from random import randrange
from math import sqrt



class Window(Frame):
    """
    The tkinter window and logic
    """

    def __init__(self, parent):
        """
        :param parent: the tkinter parent
        """

        self.parent = parent
        super(Window, self).__init__(parent)
        self.ridge_color = (0, 0, 0)
        self.initUI()

    def initUI(self):
        """
        Inits the tkinter-elements in a grid form
        """

        # ----------1. Row---------- #
        filepath_label = Label(self, text="Video file path:")
        filepath_label.grid(row=0, column=0, rowspan=1, sticky="w")

        self.path_entry = Entry(self, width=30)
        self.path_entry.grid(row=0, column=1, rowspan=1, columnspan=2, sticky="w")

        choose_btn = Button(self, text="Load Video", command=self.fileDialog)
        choose_btn.grid(row=0, column=3, rowspan=1, columnspan=1)

        # ----------2.Row---------- #
        tilenum_label = Label(self, text="Number of Mosaic Tiles:")
        tilenum_label.grid(row=1, column=0, rowspan=1, columnspan=1, sticky="w")

        self.tilenum_entry = Entry(self)
        self.tilenum_entry.insert(0, "100000")
        self.tilenum_entry.grid(row=1, column=1, rowspan=1, columnspan=1)

        self.pixel_label = Label(self, text="")
        self.pixel_label.grid(row=1, column=2, rowspan=1, columnspan=1)

        # ----------3.Row---------- #
        colorrand_label = Label(self, text="Color randomization (0.01 = 1%):")
        colorrand_label.grid(row=2, column=0, rowspan=1, columnspan=1, sticky="e")

        self.colorrand_factor = Entry(self)
        self.colorrand_factor.insert(0, "0.00")
        self.colorrand_factor.grid(row=2, column=1, rowspan=1, columnspan=1)

        # ----------4.Row---------- #
        intensity_label = Label(self, text="Color Intensity:")
        intensity_label.grid(row=3, column=0, rowspan=1, columnspan=1, sticky="w")

        self.intensity_factor_entry = Entry(self)
        self.intensity_factor_entry.insert(0, "1.0")
        self.intensity_factor_entry.grid(row=3, column=1, rowspan=1, columnspan=1)

        # ----------5.Row---------- #
        ridgeclr_label = Label(self, text="Color of Tile outline:")
        ridgeclr_label.grid(row=4, column=0, rowspan=1, columnspan=1, sticky="w")

        self.ridgeclr_display = Label(self, bg='#%02x%02x%02x' % self.ridge_color, width=10)  # Converts RGB to HEX
        self.ridgeclr_display.grid(row=4, column=1, rowspan=1, columnspan=1, sticky="w")

        ridgeclr_button = Button(self, text="Choose Color", command=self.colorChooser)
        ridgeclr_button.grid(row=4, column=2, rowspan=1, columnspan=1)

        # ----------6.Row---------- #
        create_btn = Button(self, text="Create Mosaic", command=self.createMosaic)
        create_btn.grid(row=5, column=0, rowspan=1, columnspan=1)

        self.loading_label = Label(self, text="")
        self.loading_label.grid(row=5, column=1, rowspan=1, columnspan=1)

        # ----------7. Row---------- #
        self.preview_btn = Button(self, text="Preview Image", command=self.prevImage, state="disabled")
        self.preview_btn.grid(row=6, column=1, rowspan=1, columnspan=1)

        self.save_btn = Button(self, text="Save Image", command=self.saveImage, state="disabled")
        self.save_btn.grid(row=6, column=2, rowspan=1, columnspan=1)

    def fileDialog(self):
        """
        Opens a file dialog to choose a image to convert to a mosaic
        """
        filepath = filedialog.askopenfilename(title="Select file", filetypes=[("all video format", ".mp4 .flv .avi")])
        self.path_entry.delete(0, 'end')
        self.path_entry.insert(0, filepath)

        self.video = cv2.VideoCapture(filepath)


    def colorChooser(self):
        """
        Opens a color chooser dialog and sets the ridge_color
        """
        openclr = self.ridge_color
        selected_color = colorchooser.askcolor(self.ridge_color)
        if selected_color[0]:
            self.ridge_color = tuple([int(x) for x in selected_color[0]])
            self.ridgeclr_display['bg'] = '#%02x%02x%02x' % self.ridge_color


    def createMosaic(self):
        """
        Creates a mosaic for the choosen video
        """

        tilenum_str = self.tilenum_entry.get()
        filepath_str = self.path_entry.get()

        if not isint(tilenum_str):
            self.loading_label['text'] = 'Tilenum not an integer'
            return
        if not os.path.isfile(filepath_str):
            self.loading_label['text'] = 'Invalid filepath'
            return

        self.save_btn['state'] = 'normal'
        self.save_btn.update()
        self.preview_btn['state'] = 'normal'
        self.preview_btn.update()
        self.loading_label['text'] = "Loading..."
        self.loading_label.update()

        rand_factor = float(self.colorrand_factor.get())
        intensity_factor = float(self.intensity_factor_entry.get())

        # Read Video
        ret, frame = self.video.read()
        x_size = frame.shape[1]
        y_size = frame.shape[0]

        mosaic = MosaicGenerator(x_size, y_size, int(tilenum_str), rand_factor=rand_factor,
                                 intensity_factor=intensity_factor, ridge_color=self.ridge_color)
        mosaic.calculateMosaic()

        # Video writer
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.video.get(cv2.CAP_PROP_FPS), (x_size, y_size))

        counter = 0
        while True:
            ret, frame = self.video.read()

            if not ret:
                break

            mosaic_frame = mosaic.getMosaicImage(frame)
            out.write(mosaic_frame)
            counter += 1


        self.save_btn['state'] = 'normal'
        self.preview_btn['state'] = 'normal'
        self.loading_label['text'] = "Done!"

        self.video.release()
        out.release()


    def showCanny(self):
        grayscale = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grayscale, int(self.mintreshold_entry.get()), int(self.maxtreshold_entry.get()))
        cv2.imshow("Edges", cv2.resize(edges, (800, 600)))

    def saveImage(self):
        """
        Opens a save file dialog to save the mosaic
        """
        save_path = filedialog.asksaveasfilename(title="Save Image", filetypes=[(".jpg", ".jpg")])

        temp_file = open('temp.jpg', 'rb')
        save_file = open(save_path + '.jpg', 'wb')
        save_file.write(temp_file.read())
        save_file.close()
        temp_file.close()

        self.loading_label['text'] = "Image saved!"

    def prevImage(self):
        """
        Opens a image viewer to preview the generated mosaic
        """
        imageViewerFromCommandLine = {'linux': 'xdg-open',
                                      'win32': 'explorer',
                                      'darwin': 'open'}[sys.platform]
        subprocess.run([imageViewerFromCommandLine, 'temp.jpg'])


def dist(p1, p2):
    return sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def isint(s):
    """
    Checks if string s is a integer
    :param s: string to check
    :return: True if s is a string, False otherwise
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


root = Tk(className=' Mosaic Generator')

main = Window(root)
main.pack(fill="both", expand=True)

root.mainloop()

# Remove temp file
if os.path.isfile('temp.jpg'):
    os.remove('temp.jpg')