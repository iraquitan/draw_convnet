import numpy as np
from matplotlib.patches import Rectangle

from .color import *


class Layer(object):

    def __init__(self, name, size=24, n=5, top_left=(0, 0), loc_diff=(3, -3),
                 n_show=5):
        """"""
        self.name = name
        self.size = size
        self.n = n
        self.top_left = top_left
        self.loc_diff = loc_diff
        self.n_show = n_show
        self._patches = []
        self._colors = []
        self._bottom_right = None
        self._label = None

    @property
    def label(self):
        return (self.top_left, self.name + '\n{}@{}x{}'.format(
            self.n, self.size, self.size), [0, 4])

    @label.setter
    def label(self, value):
        raise PermissionError("Not allowed to set label")

    @property
    def patches(self):
        return self._patches

    @property
    def colors(self):
        return self._colors

    @property
    def bottom_right(self):
        return self._bottom_right

    def add(self,):
        # add a rectangle
        top_left = np.array(self.top_left)
        loc_diff = np.array(self.loc_diff)
        loc_start = top_left - np.array([0, self.size])
        for ind in range(self.n_show):
            self._patches.append(
                Rectangle(loc_start + ind * loc_diff, self.size, self.size))
            if ind % 2:
                self._colors.append(Medium)
            else:
                self._colors.append(Light)
        x, y = self._patches[-1].get_xy()

        self._bottom_right = (x + self.size, y)
