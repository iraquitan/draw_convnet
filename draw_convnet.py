"""
Copyright (c) 2016, Gavin Weiguang Ding
All rights reserved.

Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""


import os
import numpy as np
import matplotlib as mpl
mpl.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt
# try:
#     import matplotlib.pyplot as plt
# except RuntimeError as e:
#     mpl.rcParams['backend'] = 'TkAgg'
#     import matplotlib.pyplot as plt
plt.rcdefaults()
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection, LineCollection

NumConvMax = 8
NumFcMax = 20
White = 1.
Light = 0.7
Medium = 0.5
Dark = 0.3
Black = 0.
my_cmap = plt.cm.ocean


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


def add_mapping_2(lines, start, end):
    for start_loc, end_loc in zip(start, end):
        lines.append([start_loc, end_loc])


def label(xy, text, xy_off=[0, 4]):
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
             family='sans-serif', size=8)


if __name__ == '__main__':

    fc_unit_size = 2
    layer_width = 40

    patches = []
    lines = []
    lines_bg = []
    colors = []

    fig, ax = plt.subplots()


    ############################
    # conv layers
    size_list = [32, 18, 10, 6, 4]
    num_list = [3, 32, 32, 48, 48]
    x_diff_list = [0, layer_width, layer_width, layer_width, layer_width]
    text_list = ['Inputs'] + ['Feature\nmaps'] * (len(size_list) - 1)
    loc_diff_list = [[3, -3]] * len(size_list)

    num_show_list = list(map(min, num_list, [NumConvMax] * len(num_list)))
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]

    bg_line_start = []
    bg_line_end = []
    bg_line_start.append(top_left_list[-1])

    for ind in range(len(size_list)):
        add_layer(patches, colors, size=size_list[ind],
                  num=num_show_list[ind],
                  top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}@{}x{}'.format(
            num_list[ind], size_list[ind], size_list[ind]))

    bg_line_start.append(patches[-1].get_xy())

    ############################
    # in between layers
    start_ratio_list = [[0.4, 0.5], [0.4, 0.8], [0.4, 0.5], [0.4, 0.8]]
    patch_size_list = [5, 2, 5, 2]
    ind_bgn_list = range(len(patch_size_list))
    text_list = ['Convolution', 'Max-pooling', 'Convolution', 'Max-pooling']

    for ind in range(len(patch_size_list)):
        add_mapping(patches, lines, colors, start_ratio_list[ind],
                    patch_size_list[ind], ind,
                    top_left_list, loc_diff_list, num_show_list, size_list)
        label(top_left_list[ind], text_list[ind] + '\n{}x{} kernel'.format(
            patch_size_list[ind], patch_size_list[ind]), xy_off=[26, -65])


    ############################
    # fully connected layers
    size_list = [fc_unit_size, fc_unit_size, fc_unit_size]
    num_list = [768, 500, 2]
    num_show_list = list(map(min, num_list, [NumFcMax] * len(num_list)))
    x_diff_list = [sum(x_diff_list) + layer_width, layer_width, layer_width]
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]
    loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)
    text_list = ['Hidden\nunits'] * (len(size_list) - 1) + ['Outputs']

    bg_line_end.append(top_left_list[0])

    for ind in range(len(size_list)):
        if ind > 0:
            tx, ty = patches[-1].get_xy()
            h, w = patches[-1].get_height(), patches[-1].get_width()
            bg_line_start.append((tx + w, ty))
            # my_top_left_start.append(patches[-1].get_xy())
        add_layer(patches, colors, size=size_list[ind], num=num_show_list[ind],
                  top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}'.format(
            num_list[ind]))

        tx, ty = patches[-1].get_xy()
        h, w = patches[-1].get_height(), patches[-1].get_width()
        bg_line_end.append((tx + w, ty))
    text_list = ['Flatten\n', 'Fully\nconnected', 'Fully\nconnected']

    bg_line_start.append(top_left_list[0])
    bg_line_end.append(top_left_list[1])
    bg_line_start.append(top_left_list[1])
    bg_line_end.append(top_left_list[2])

    add_mapping_2(lines_bg, bg_line_start, bg_line_end)

    for ind in range(len(size_list)):
        label(top_left_list[ind], text_list[ind], xy_off=[-10, -65])

    ############################
    colors += [0, 1]
    linebg_collection = LineCollection(lines_bg, colors='k', linewidths=0.5,
                                       zorder=1)
    ax.add_collection(linebg_collection)
    collection = PatchCollection(patches, cmap=my_cmap)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)
    line_collection = LineCollection(lines, colors='k', linewidths=0.5)
    ax.add_collection(line_collection)
    plt.tight_layout()
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    fig.set_size_inches(8, 2.5)

    fig_dir = './'
    fig_ext = '.svg'
    fig.savefig(os.path.join(fig_dir, 'convnet_fig1' + fig_ext),
                bbox_inches='tight', pad_inches=0)
