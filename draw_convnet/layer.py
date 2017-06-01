import numpy as np
from matplotlib.patches import Rectangle

from draw_convnet.color import *


def add_layer(patches, colors, size=24, num=5, top_left=[0, 0],
              loc_diff=[3, -3]):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size])
    for ind in range(num):
        patches.append(Rectangle(loc_start + ind * loc_diff, size, size))
        if ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)
