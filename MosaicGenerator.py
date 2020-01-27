import os
import sys
import cv2
import pygame
import subprocess
import multiprocessing
import numpy as np

from tkinter import Tk, Frame, Button, Label, Entry
from tkinter import filedialog
from scipy.spatial import Voronoi
from random import randrange
from math import sqrt
from functools import partial


class MosaicGenerator:
    """
    Generates a Mosaic based on a given image and the number of tiles using the Voronoi algorithm.
    Every Voronoi region is a Mosaic tile
    """

    def __init__(self, img_path, num_tiles):
        """
        :param img_path: file path to the image
        :param num_tiles: number of mosaic tiles to generate
        """

        self.img_path = img_path
        self.img = cv2.imread(img_path)
        # get dimensions of the image
        self.x_size = self.img.shape[1]
        self.y_size = self.img.shape[0]

        # edge detection
        #edges = cv2.Canny(self.img_path, 100, 200)
        #cv2.imshow("Edges", edges)

        self.num_tiles = num_tiles

        # assign random points on the image
        self.points = np.array(
            [[randrange(-int(self.x_size * 0.05), int(self.x_size * 1.05)),
              randrange(-int(self.y_size * 0.05), int(self.y_size * 1.05))]])

        self.resetPoints()

    def resetPoints(self):
        """
        Resets the current points of the mosaic
        TODO: Better point distribution
        """
        r_x = int(self.x_size * 1.1)
        r_y = int(self.y_size * 1.1)

        while len(self.points) < self.num_tiles:
            x = randrange(-int(self.x_size * 0.05), int(self.x_size * 1.05))
            y = randrange(-int(self.y_size * 0.05), int(self.y_size * 1.05))

            self.points = np.append(self.points, [[x, y]], axis=0)


    def calculateMosaic(self):
        """
        Calculates the Voronoi based on the assigned points
        """

        vor = Voronoi(self.points)
        self.points = vor.points
        self.verts = vor.vertices
        self.ridge_vertices = vor.ridge_vertices
        self.regions = vor.regions

    def getAverageRegionColor(self, px, py, num_neighbor_pixel=1):
        """
        Returns the average pixel color of a pixel and its surrounding pixels
        :param px: the x value of the pixel
        :param py: the y value of the pixel
        :param num_neighbor_pixel: number of neighbour pixels to consider (=1 means 9 pixels get considered)
        :return: the average color of the considered pixels [r, g, b]
        """

        clrsum = [0, 0, 0]

        count = 0
        for x in range(px-num_neighbor_pixel, px+num_neighbor_pixel+1):
            for y in range(py-num_neighbor_pixel, py+num_neighbor_pixel+1):
                # Check if the pixel is in the image
                if x >= 0 and x < self.x_size and y >= 0 and y < self.y_size:
                    clrsum[0] += self.img[y][x][2]
                    clrsum[1] += self.img[y][x][1]
                    clrsum[2] += self.img[y][x][0]
                    count += 1

        if count != 0:
            for i in range(len(clrsum)):
                clrsum[i] /= count
        else:
            None
            #TODO: Randteile richtig einfärben..

        return clrsum

    def drawRegions(self):
        """
        Draws all voronoi regions as pygame polygons
        """

        for r in self.regions:
            if -1 not in r and len(r) > 2:

                polygon_points = []
                midpoint = [0, 0]

                for v in r:
                    x = self.verts[v][0]
                    y = self.verts[v][1]
                    polygon_points.append([x, y])
                    midpoint[0] += x
                    midpoint[1] += y

                # calculate middle point between all vertices
                midpoint[0] = int(midpoint[0] / len(r))
                midpoint[1] = int(midpoint[1] / len(r))

                # get the pixel color of this midpoint
                clr = self.getAverageRegionColor(midpoint[0], midpoint[1], num_neighbor_pixel=1)
                # draw the pygame polygon
                pygame.draw.polygon(self.screen, clr, polygon_points)

    def drawRidges(self, clr=(0, 0, 0)):
        """
        Draws the ridges between the voronoi vertices as pygame aalines -> the outline of the mosaic tiles
        :param clr: the color of the ridge line
        """

        for i in range(len(self.ridge_vertices)):
            r = self.ridge_vertices[i]

            p1 = self.verts[r[0]]
            p2 = self.verts[r[1]]
            start_point = [int(p1[0]), int(p1[1])]
            end_point = [int(p2[0]), int(p2[1])]

            if r[0] != -1:
                pygame.draw.aaline(self.screen, clr, start_point, end_point)

    def saveTempImage(self):
        """
        Saves the generated pygame mosaic image as temporary image (temp.jpg)
        """
        pygame.init()
        pygame.display.iconify()
        self.screen = pygame.display.set_mode([self.x_size, self.y_size], display=pygame.FULLSCREEN)
        pyg_img = pygame.image.load(self.img_path)
        self.screen.blit(pyg_img, [0, 0])

        self.drawRegions()
        self.drawRidges()

        pygame.display.flip()
        pygame.image.save(self.screen, "temp.jpg")
        pygame.quit()


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
        self.initUI()

    def initUI(self):
        """
        Inits the tkinter-elements in a grid form
        """

        #----------1. Row----------#
        filepath_label = Label(self, text="Image file path:")
        filepath_label.grid(row=0, column=0, rowspan=1)

        self.path_entry = Entry(self, width=30)
        self.path_entry.grid(row=0, column=1, rowspan=1, columnspan=2)

        choose_btn = Button(self, text="Load Image", command=self.fileDialog)
        choose_btn.grid(row=0, column=3, rowspan=1, columnspan=1)
        '''
        #----------2.Row----------#
        self.mintreshold_entry = Entry(self)
        self.mintreshold_entry.insert(0, "100")
        self.mintreshold_entry.grid(row=1, column=0, rowspan=1, columnspan=1)

        self.maxtreshold_entry = Entry(self)
        self.maxtreshold_entry.insert(0, "200")
        self.maxtreshold_entry.grid(row=1, column=1, rowspan=1, columnspan=1)

        self.showcanny_button = Button(self, text="show_edges", command=self.showCanny)
        self.showcanny_button.grid(row=1, column=2, rowspan=1, columnspan=1)
        '''
        #----------3.Row----------#
        tilenum_label = Label(self, text="Number of Mosaic Tiles:")
        tilenum_label.grid(row=2, column=0, rowspan=1, columnspan=1)

        self.tilenum_entry = Entry(self)
        self.tilenum_entry.insert(0, "100000")
        self.tilenum_entry.grid(row=2, column=1, rowspan=1, columnspan=1)

        self.pixel_label = Label(self, text="")
        self.pixel_label.grid(row=2, column=2, rowspan=1, columnspan=1)

        #----------4.Row----------#
        create_btn = Button(self, text="Create Mosaic", command=self.createMosaic)
        create_btn.grid(row=3, column=0, rowspan=1, columnspan=1)

        self.loading_label = Label(self, text="")
        self.loading_label.grid(row=3, column=1, rowspan=1, columnspan=1)

        #----------5. Row----------#
        self.preview_btn = Button(self, text="Preview Image", command=self.prevImage, state="disabled")
        self.preview_btn.grid(row=4, column=1, rowspan=1, columnspan=1)

        self.save_btn = Button(self, text="Save Image", command=self.saveImage, state="disabled")
        self.save_btn.grid(row=4, column=2, rowspan=1, columnspan=1)

    def fileDialog(self):
        """
        Opens a file dialog to choose a image to convert to a mosaic
        """
        filepath = filedialog.askopenfilename(title="Select file", filetypes=[("Image files", ".jpg .png")])
        self.path_entry.delete(0, 'end')
        self.path_entry.insert(0, filepath)

        self.img = cv2.imread(filepath)
        # get dimensions of the image
        x_size = self.img.shape[1]
        y_size = self.img.shape[0]

        self.pixel_label['text'] = "Pixels: " + str(x_size) + " | " + str(y_size)\
                                   + " = " + str(x_size*y_size)


    def createMosaic(self):
        """
        Creates a mosaic for the choosen image
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

        mosaic = MosaicGenerator(filepath_str, int(tilenum_str))
        mosaic.calculateMosaic()
        mosaic.saveTempImage()

        self.save_btn['state'] = 'normal'
        self.preview_btn['state'] = 'normal'
        self.loading_label['text'] = "Done!"

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
    return sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


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