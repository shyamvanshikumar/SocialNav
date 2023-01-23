#!/usr/bin/env python3

import numpy as np
from termcolor import cprint

class BEVLidar:
    def __init__(self, x_range=(-20, 20),
                 y_range=(-20, 20),
                 z_range=(-1, 5),
                 resolution=0.05,
                 threshold_z_range=False):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.resolution = resolution
        self.dx = x_range[1]/resolution
        self.dy = y_range[1]/resolution
        self.img_size = int(1 + (x_range[1] - x_range[0]) / resolution)
        self.threshold_z_range = threshold_z_range
        cprint('created the bev image handler class', 'green', attrs=['bold'])

    def get_bev_lidar_img(self, lidar_points):
        img = np.zeros((self.img_size, self.img_size))
        for x, y, z in lidar_points:
            if self.not_in_range_check(x, y, z): continue
            ix = (self.dx + int(x / self.resolution))
            iy = (self.dy - int(y / self.resolution))
            if self.threshold_z_range:
                img[int(round(iy)), int(round(ix))] = 1 if z >= self.z_range[0] else 0
            else:
                img[int(round(iy)), int(round(ix))] = (((z-self.z_range[0])/float(self.z_range[1]-self.z_range[0])) * 255).astype(np.uint8)
        return img

    def not_in_range_check(self, x, y, z):
        if x < self.x_range[0] \
                or x > self.x_range[1] \
                or y < self.y_range[0] \
                or y > self.y_range[1] \
                or z < self.z_range[0] \
                or z > self.z_range[1]: return True
        return False
