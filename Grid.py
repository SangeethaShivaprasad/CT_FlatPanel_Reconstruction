import numpy as np
import flat_panel_project_utils as utils

class Grid:
    def __init__(self, height, width, spacing):
        self.height = height
        self.width = width
        self.spacing = spacing
        self.origin = (-(0.5 * (self.height - 1) * self.spacing[0])), (-(0.5 * (self.width - 1) * self.spacing[1]))
        self.buffer = np.zeros((self.height, self.width))

    def set_buffer(self, buf):
        self.buffer = buf

    def get_buffer(self):
        return self.buffer

    def get_origin(self):
        return self.origin

    def set_origin(self, origin):
        self.origin = origin

    def get_spacing(self):
        return self.spacing

    def get_size(self):
        size = (self.height, self.width)
        return size

    def index_to_physical(self,i,j):
        physical_index = (self.origin[0] + i*self.spacing[0], self.origin[1] + j*self.spacing[1])
        return physical_index

    def physical_to_index(self, x, y):
        pixel_index = ((x - self.origin[0]) / self.spacing[0], (y - self.origin[1]) / self.spacing[1])
        return pixel_index

    def set_at_index(self, i, j, val):
        self.buffer[i, j] = val

    def get_at_index(self, i, j):
        return self.buffer[i, j]

    def get_at_physical(self, x, y):
        physical_index = []
        physical_index = self.physical_to_index(x, y)
        empty_arr = utils.interpolate(self, physical_index[0], physical_index[1])
        return empty_arr
