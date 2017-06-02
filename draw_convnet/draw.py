import os
import matplotlib as mpl
mpl.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt
# try:
#     import matplotlib.pyplot as plt
# except RuntimeError as e:
#     mpl.rcParams['backend'] = 'TkAgg'
#     import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection

from .layer import add_layer
from .mapping import add_mapping, add_mapping2
from .label import label


class ConvNet(object):
    ConvMaxN = 8
    FcMaxN = 20

    def __init__(self, conv_layers_sizes, conv_layers_n, mappings_sizes,
                 mappings_labels, mappings_st_ratio, fcs_sizes, fcs_labels,
                 fcs_n, fc_unit_size=2, layer_width=40):
        """"""
        self.conv_layers_sizes = conv_layers_sizes
        self.conv_layers_n = conv_layers_n
        self.mappings_sizes = mappings_sizes
        self.mappings_labels = mappings_labels
        self.mappings_st_ratio = mappings_st_ratio
        self.fcs_sizes = fcs_sizes
        self.fcs_labels = fcs_labels
        self.fcs_n = fcs_n
        self.fc_unit_size = fc_unit_size
        self.layer_width = layer_width
        self._patches = []
        self._lines = []
        self._bg_lines = []
        self._bg_lines_st = []
        self._bg_lines_end = []
        self._colors = []
        self._labels = []
        self._x_diff = None
        self._loc_diff = None
        self._num_show = None
        self._top_left = None
        self._fig = None
        self._ax = None

    def add_conv_layers(self):
        self._x_diff = [0] + [self.layer_width for _ in
                              range(len(self.conv_layers_sizes) - 1)]
        text_list = ['Inputs'] + ['Feature\nmaps'] * (
            len(self.conv_layers_sizes) - 1)
        self._loc_diff = [[3, -3]] * len(self.conv_layers_sizes)
        self._num_show = list(map(min, self.conv_layers_n,
                                  [self.ConvMaxN] * len(self.conv_layers_n)))
        self._top_left = np.c_[
            np.cumsum(self._x_diff), np.zeros(len(self._x_diff))]

        # self._bg_lines_st = []
        # self._bg_lines_end = []
        self._bg_lines_st.append(self._top_left[-1])

        for ind in range(len(self.conv_layers_sizes)):
            add_layer(self._patches, self._colors,
                      size=self.conv_layers_sizes[ind],
                      num=self._num_show[ind], top_left=self._top_left[ind],
                      loc_diff=self._loc_diff[ind])
            tl = {'xy': self._top_left[ind],
                  'text': text_list[ind] + '\n{}@{}x{}'.format(
                      self.conv_layers_n[ind], self.conv_layers_sizes[ind],
                      self.conv_layers_sizes[ind])}
            self._labels.append(tl)

        self._bg_lines_st.append(self._patches[-1].get_xy())

    def add_mappings(self):
        text_list = ['Convolution', 'Max-pooling', 'Convolution',
                     'Max-pooling']

        for ind in range(len(self.mappings_sizes)):
            add_mapping(self._patches, self._lines, self._colors,
                        self.mappings_st_ratio[ind],
                        self.mappings_sizes[ind], ind,
                        self._top_left, self._loc_diff, self._num_show,
                        self.conv_layers_sizes)

            tl = {'xy': self._top_left[ind],
                  'text': text_list[ind] + '\n{}x{} kernel'.format(
                    self.mappings_sizes[ind], self.mappings_sizes[ind]),
                  'xy_off': [26, -65]}
            self._labels.append(tl)

    def add_fc_layers(self):
        self._num_show = list(map(min, self.fcs_n,
                                  [self.FcMaxN] * len(self.fcs_n)))
        self._x_diff = [sum(self._x_diff) + self.layer_width] + [
            self.layer_width for _ in range(len(self.fcs_sizes) - 1)]
        self._top_left = np.c_[
            np.cumsum(self._x_diff), np.zeros(len(self._x_diff))]
        self._loc_diff = [[self.fc_unit_size, -self.fc_unit_size]] * len(
            self._top_left)
        text_list = ['Hidden\nunits'] * (len(self.fcs_sizes) - 1) + ['Outputs']

        self._bg_lines_end.append(self._top_left[0])

        for ind in range(len(self.fcs_sizes)):
            if ind > 0:
                tx, ty = self._patches[-1].get_xy()
                h, w = self._patches[-1].get_height(), self._patches[
                    -1].get_width()
                self._bg_lines_st.append((tx + w, ty))
            add_layer(self._patches, self._colors, size=self.fcs_sizes[ind],
                      num=self._num_show[ind],
                      top_left=self._top_left[ind],
                      loc_diff=self._loc_diff[ind])

            tl = {'xy': self._top_left[ind],
                  'text': text_list[ind] + '\n{}'.format(self.fcs_n[ind])}
            self._labels.append(tl)
            tx, ty = self._patches[-1].get_xy()
            h, w = self._patches[-1].get_height(), self._patches[
                -1].get_width()
            self._bg_lines_end.append((tx + w, ty))

        self._bg_lines_st.append(self._top_left[0])
        self._bg_lines_end.append(self._top_left[1])
        self._bg_lines_st.append(self._top_left[1])
        self._bg_lines_end.append(self._top_left[2])

        add_mapping2(self._bg_lines, self._bg_lines_st, self._bg_lines_end)

        for ind in range(len(self.fcs_sizes)):
            tl = {'xy': self._top_left[ind],
                  'text': self.fcs_labels[ind], 'xy_off': [-10, -65]}
            self._labels.append(tl)

    def plot(self):
        self._fig, self._ax = plt.subplots()
        self._colors += [0, 1]
        linebg_collection = LineCollection(self._bg_lines, colors='k',
                                           linewidths=0.5,
                                           zorder=1)
        self._ax.add_collection(linebg_collection)
        collection = PatchCollection(self._patches, cmap=mpl_cm.ocean)
        collection.set_array(np.array(self._colors))
        self._ax.add_collection(collection)
        line_collection = LineCollection(self._lines, colors='k',
                                         linewidths=0.5)
        self._ax.add_collection(line_collection)

        for l in self._labels:
            label(axes=self._ax, **l)

        plt.tight_layout()
        plt.axis('equal')
        plt.axis('off')
        plt.show()

    def save_plot(self):
        fig_dir = './'
        fig_ext = '.svg'
        self._fig.set_size_inches(8, 2.5)
        self._fig.savefig(os.path.join(fig_dir, 'convnet_fig1' + fig_ext),
                          bbox_inches='tight', pad_inches=0)
