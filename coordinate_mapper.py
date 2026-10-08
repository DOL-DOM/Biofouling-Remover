import numpy as np

class CoordinateMapper:
    def __init__(self, frame_width=640, frame_height=480, grid_size=20):
        self.fw = frame_width
        self.fh = frame_height
        self.grid_size = grid_size

    def pixel_to_grid(self, pixel_x, pixel_y):
        gx = int(np.clip((pixel_x / self.fw) * self.grid_size, 0, self.grid_size - 1))
        gy = int(np.clip((pixel_y / self.fh) * self.grid_size, 0, self.grid_size - 1))
        return gx, gy