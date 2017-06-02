from matplotlib.patches import Rectangle
import numpy as np
from .color import *


class Map(object):
    def __init__(self, name, from_, to_, size=24, start_ratio=[0.4, 0.5]):
        """"""
        self.name = name
        self.from_ = from_
        self.to_ = to_
        self.size = size
        self.start_ratio = start_ratio
        self._patches = []
        self._lines = []
        self._colors = []
        self._bottom_right = None
        self._label = None

    @property
    def label(self):
        return (self.from_.top_left, self.name + '\n{}x{}'.format(
            self.size, self.size), [26, -65])

    @label.setter
    def label(self, value):
        raise PermissionError("Not allowed to set label")

    @property
    def patches(self):
        return self._patches

    @property
    def lines(self):
        return self._lines

    @property
    def colors(self):
        return self._colors

    def add(self,):
        start_loc = self.from_.top_left \
                    + (self.from_.n - 1) * np.array(self.from_.loc_diff) \
                    + np.array([self.start_ratio[0] * self.from_.size,
                                -self.start_ratio[1] * self.from_.size])

        end_loc = self.to_.top_left \
                  + (self.to_.n - 1) * np.array(self.to_.loc_diff) \
                  + np.array([(self.start_ratio[0] + .5 *
                               self.size / self.from_.size) * self.to_.size,
                              -(self.start_ratio[1] - .5 *
                                self.size / self.from_.size) * self.to_.size])

        self._patches.append(Rectangle(start_loc, self.size, self.size))
        self._colors.append(Dark)
        self._lines.append([start_loc, end_loc])
        self._lines.append([(start_loc[0] + self.size, start_loc[1]), end_loc])
        self._lines.append([(start_loc[0], start_loc[1] + self.size), end_loc])
        self._lines.append([(start_loc[0] + self.size,
                             start_loc[1] + self.size), end_loc])
