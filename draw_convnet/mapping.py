import numpy as np
from matplotlib.patches import Rectangle

from draw_convnet.color import *


def add_mapping(patches, lines, colors, start_ratio, patch_size, ind_bgn,
                top_left_list, loc_diff_list, num_show_list, size_list):

    start_loc = top_left_list[ind_bgn] \
        + (num_show_list[ind_bgn] - 1) * np.array(loc_diff_list[ind_bgn]) \
        + np.array([start_ratio[0] * size_list[ind_bgn],
                    -start_ratio[1] * size_list[ind_bgn]])

    end_loc = top_left_list[ind_bgn + 1] \
        + (num_show_list[ind_bgn + 1] - 1) \
        * np.array(loc_diff_list[ind_bgn + 1]) \
        + np.array([(start_ratio[0] + .5 * patch_size / size_list[ind_bgn]) *
                    size_list[ind_bgn + 1],
                    -(start_ratio[1] - .5 * patch_size / size_list[ind_bgn]) *
                    size_list[ind_bgn + 1]])

    patches.append(Rectangle(start_loc, patch_size, patch_size))
    colors.append(Dark)
    lines.append([start_loc, end_loc])
    lines.append([(start_loc[0] + patch_size, start_loc[1]),
                  (end_loc[0], end_loc[1])])
    lines.append([(start_loc[0], start_loc[1] + patch_size),
                  (end_loc[0], end_loc[1])])
    lines.append([(start_loc[0] + patch_size, start_loc[1] + patch_size),
                  (end_loc[0], end_loc[1])])
