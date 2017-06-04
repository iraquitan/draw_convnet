import os

import matplotlib as mpl
mpl.rcParams['backend'] = 'TkAgg'
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import numpy as np

from .layer import Layer
from .mapping import Map, MapFC


class BaseNet(object):
    def __init__(self, name, layer_width=40, conv_max_n=8, fc_max_n=20,
                 fc_unit_size=2, cmap='ocean', line_width=0.25):
        """"""
        self.name = name
        self.layer_width = layer_width
        self.conv_max_n = conv_max_n
        self.fc_max_n = fc_max_n
        self.fc_unit_size = fc_unit_size
        self.cmap = cmap
        self.line_width = line_width
        self._conv_layers = []
        self._mappings = []
        self._mappings_fc = []
        self._fc_layers = []
        self._fig = None
        self._ax = None

    @property
    def get_n_conv_layers(self):
        return len(self._conv_layers)

    @property
    def get_n_fc_layers(self):
        return len(self._fc_layers)

    @property
    def get_n_mappings(self):
        return len(self._mappings)

    @property
    def patches(self):
        patches = [
            i.patches for l in [self._conv_layers, self._mappings,
                                self._fc_layers]
            for i in l
        ]
        return [item for sublist in patches for item in sublist]

    @property
    def colors(self):
        colors = [
            i.colors for l in [self._conv_layers, self._mappings,
                               self._fc_layers]
            for i in l
        ]
        return [item for sublist in colors for item in sublist]

    @property
    def lines(self):
        lines = [i.lines for i in self._mappings]
        return [item for sublist in lines for item in sublist]

    @property
    def lines_bg(self):
        lines = [i.lines for i in self._mappings_fc]
        return [item for sublist in lines for item in sublist]

    @property
    def labels(self):
        labels = [
            i.label for l in [self._conv_layers, self._mappings,
                              self._mappings_fc, self._fc_layers]
            for i in l
        ]
        # return [item for sublist in labels for item in sublist]
        return labels

    def add_conv_layer(self, name, size, n):
        # x_diff = self.layer_width if self._conv_layers else 0
        loc_diff = [3, -3]
        top_left = (self.layer_width * len(self._conv_layers), 0)
        n_show = self.conv_max_n if n > self.conv_max_n else n
        layer = Layer(name, 'conv', size, n, top_left, loc_diff, n_show)
        layer.add()
        self._conv_layers.append(layer)

    def add_fc_layer(self, name, n):
        # x_diff = self.layer_width if self._conv_layers else 0
        loc_diff = [self.fc_unit_size, -self.fc_unit_size]
        top_left = (self.layer_width * (
            len(self._conv_layers) + len(self._fc_layers)), 0)
        n_show = self.fc_max_n if n > self.fc_max_n else n
        layer = Layer(name, 'fc', self.fc_unit_size, n, top_left, loc_diff,
                      n_show)
        layer.add()
        self._fc_layers.append(layer)

    def add_conv_mapping(self, name, from_ix, to_ix, size, start_ratio):
        if self.get_n_conv_layers % 2 == 0:
            raise RuntimeError("To add a convolution mapping, the number of "
                               "convolutional layers must be odd.")
        from_layer = self._conv_layers[from_ix]
        to_layer = self._conv_layers[to_ix]
        map = Map(name, from_layer, to_layer, size, start_ratio)
        map.add()
        self._mappings.append(map)

    def _add_fc_mapping(self, name, from_layer, to_layer, last=False):
        map = MapFC(name, from_layer, to_layer)
        map.add(last)
        self._mappings_fc.append(map)

    def add_label(self, xy, text, xy_off=[0, 4]):
        self._ax.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
                      family='sans-serif', size=8)

    def draw(self, show=False):
        self._fig, self._ax = plt.subplots()
        colors = self.colors
        colors += [0, 1]
        # Add fc_mappings
        self._add_fc_mapping('Flatten\n', self._conv_layers[-1],
                             self._fc_layers[0])
        for i in range(len(self._fc_layers) - 1):
            last = False
            if i == len(self._fc_layers) - 2:
                # last = True
                last = False
            self._add_fc_mapping('Fully\nconnected', self._fc_layers[i],
                                 self._fc_layers[i+1], last)

        linebg_collection = LineCollection(self.lines_bg, colors='k',
                                           linewidths=self.line_width,
                                           zorder=1)
        self._ax.add_collection(linebg_collection)

        cmap = mpl_cm.get_cmap(self.cmap)
        collection = PatchCollection(self.patches, cmap=cmap)
        collection.set_array(np.array(colors))
        self._ax.add_collection(collection)
        line_collection = LineCollection(self.lines, colors='k',
                                         linewidths=self.line_width)
        self._ax.add_collection(line_collection)

        for l in self.labels:
            self.add_label(*l)

        plt.tight_layout()
        plt.axis('equal')
        plt.axis('off')
        if show:
            plt.show()

    def save_plot(self):
        fig_dir = './'
        fig_ext = '.svg'
        self._fig.set_size_inches(8, 2.5)
        self._fig.savefig(os.path.join(fig_dir, 'dnn_draw_fig' + fig_ext),
                          bbox_inches='tight', pad_inches=0)
