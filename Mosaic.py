import os
import sys
import cv2
import pygame
import subprocess
import numpy as np

from tkinter import Tk, Frame, Button, Label, Entry
from tkinter import filedialog, colorchooser
from scipy.spatial import Voronoi
from random import randrange
from math import sqrt


class MosaicGenerator:
    """
    Generates a Mosaic based on a given image and the number of tiles using the Voronoi algorithm.
    Every Voronoi region is a Mosaic tile
    """

    def __init__(self, x_size, y_size, num_tiles, rand_factor=0.0, intensity_factor=1.0, ridge_color=(0,0,0)):
        """
        :param img_path: file path to the image
        :param num_tiles: number of mosaic tiles to generate
        """
        # get dimensions of the image
        self.x_size = x_size
        self.y_size = y_size

        # edge detection
        # edges = cv2.Canny(self.img_path, 100, 200)
        # cv2.imshow("Edges", edges)

        self.num_tiles = num_tiles
        self.rand_factor = rand_factor
        self.intensity_factor = intensity_factor
        self.ridge_color = ridge_color

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

    def getPixelRGB(self, x, y):
        """
        Returns the rgb of a certain pixel on the image
        :param x: x coordinate of the pixel
        :param y: y coordinate of the pixel
        :return: the pixel color(rgb) or None if coordinates outside of the image
        """
        if 0 <= x < self.x_size and 0 <= y < self.y_size:
            color = None
            try:
                color = self.img[y][x]
            except:
                return None
            r = color[2]
            g = color[1]
            b = color[0]
            return [r, g, b]
        else:
            return None

    def getAverageRegionColor(self, px, py, region_verts, num_neighbor_pixel=1):
        """
        Returns the average pixel color of a pixel and its surrounding pixels
        :param px: the x value of the pixel
        :param py: the y value of the pixel
        :param region_verts: the vertices of the region, in case midpoint is not in the image
        :param num_neighbor_pixel: number of neighbour pixels to consider (=1 means 9 pixels get considered)
        :return: the average color of the considered pixels [r, g, b]
        """
        clrsum = [0, 0, 0]

        count = 0
        for x in range(px - num_neighbor_pixel, px + num_neighbor_pixel + 1):
            for y in range(py - num_neighbor_pixel, py + num_neighbor_pixel + 1):
                pixel_clr = self.getPixelRGB(x, y)
                if pixel_clr:
                    clrsum[0] += pixel_clr[0]
                    clrsum[1] += pixel_clr[1]
                    clrsum[2] += pixel_clr[2]
                    count += 1

        if count != 0:
            for i in range(len(clrsum)):
                clrsum[i] /= count
                clrsum[i] *= self.intensity_factor

        else:
            # px, py and neighbours not in image -> get colors from vertices
            for v in region_verts:
                x = self.verts[v][0]
                y = self.verts[v][1]
                pixel_clr = self.getPixelRGB(x, y)
                # TODO: Warum immer None?????
                if pixel_clr:
                    clrsum[0] += pixel_clr[0]
                    clrsum[1] += pixel_clr[1]
                    clrsum[2] += pixel_clr[2]
                    count += 1

            if count != 0:
                for i in range(len(clrsum)):
                    clrsum[i] /= count
                    clrsum[i] *= self.intensity_factor

        for i in range(len(clrsum)):
            if clrsum[i] > 255:
                clrsum[i] = 255

        return clrsum


    def randomizeColor(self, clr):
        """
        Randomizes the given color(clr) by self.rand_factor
        TODO: in helper module auslagern
        """

        if self.rand_factor == 0.0:
            return clr

        val = 255 * self.rand_factor
        for i in range(len(clr)):
            start = int(clr[i] - val)
            end = int(clr[i] + val)
            if start < 0:
                start = 0
            if end > 256:
                end = 256
            clr[i] = randrange(start, end)
        return clr

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
                clr = self.getAverageRegionColor(midpoint[0], midpoint[1], r, num_neighbor_pixel=1)
                # randomize color
                clr = self.randomizeColor(clr)
                # draw the pygame polygon
                pygame.draw.polygon(self.screen, clr, polygon_points)

    def drawRidges(self):
        """
        Draws the ridges between the voronoi vertices as pygame aalines -> the outline of the mosaic tiles
        Color used: self.ridge_color
        """

        for i in range(len(self.ridge_vertices)):
            r = self.ridge_vertices[i]

            p1 = self.verts[r[0]]
            p2 = self.verts[r[1]]
            start_point = [int(p1[0]), int(p1[1])]
            end_point = [int(p2[0]), int(p2[1])]

            if r[0] != -1:
                pygame.draw.aaline(self.screen, self.ridge_color, start_point, end_point)

    def saveTempImage(self, image, path):
        """
        Saves the generated pygame mosaic image as temporary image (temp.jpg)
        """

        self.img = image
        cv2.imwrite("temp_" + path, image)

        pygame.init()
        pygame.display.iconify()
        self.screen = pygame.display.set_mode([self.x_size, self.y_size], display=pygame.FULLSCREEN)
        pyg_img = pygame.image.load("temp_" + path)
        self.screen.blit(pyg_img, [0, 0])

        self.drawRegions()
        self.drawRidges()

        pygame.display.flip()
        pygame.image.save(self.screen, path)
        pygame.quit()

    def getMosaicImage(self, image):
        self.img = image
        cv2.imwrite("temp.jpg", image)

        pygame.init()
        pygame.display.iconify()
        self.screen = pygame.display.set_mode([self.x_size, self.y_size], display=pygame.FULLSCREEN)
        pyg_img = pygame.image.load("temp.jpg")
        self.screen.blit(pyg_img, [0, 0])

        self.drawRegions()
        self.drawRidges()

        pygame.display.flip()

        color_image = pygame.surfarray.array3d(self.screen)

        color_image = cv2.transpose(color_image)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        pygame.quit()

        return color_image


    def cvimage_to_pygame(self, image):
        """Convert cvimage into a pygame image"""
        return pygame.image.frombuffer(image.tostring(), image.shape[1::-1], "RGB")