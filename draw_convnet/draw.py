import numpy as np
from .layer import add_layer
from .mapping import add_mapping
from .label import label


class ConvNet(object):
    ConvMaxN = 8
    FcMaxN = 20

    def __init__(self, conv_layers_sizes, conv_layers_n, mappings_sizes,
                 mappings_labels, fcs_sizes, fcs_n, fc_unit_size=2,
                 layer_width=40):
        """"""
        self.conv_layers_sizes = conv_layers_sizes
        self.conv_layers_n = conv_layers_n
        self.mappings_sizes = mappings_sizes
        self.mappings_labels = mappings_labels
        self.fcs_sizes = fcs_sizes
        self.fcs_n = fcs_n
        self.fc_unit_size = fc_unit_size
        self.layer_width = layer_width
        self._patches = []
        self._lines = []
        self._colors = []
        self._loc_diff = None
        self._num_show = None
        self._top_left = None

    def add_conv_layers(self):
        x_diff_list = [0] + [self.layer_width for i in self.conv_layers_sizes]
        text_list = ['Inputs'] + ['Feature\nmaps'] * (
            len(self.conv_layers_sizes) - 1)
        self._loc_diff = [[3, -3]] * len(self.conv_layers_sizes)

        self._num_show = list(map(min, self.conv_layers_n, [self.ConvMaxN] * len(self.conv_layers_n)))
        self._top_left = np.c_[
            np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]

        # bg_line_start = []
        # bg_line_end = []
        # bg_line_start.append(top_left_list[-1])

        for ind in range(len(self.conv_layers_sizes)):
            add_layer(self._patches, self._colors,
                      size=self.conv_layers_sizes[ind], num=self.num_show[ind],
                      top_left=self._top_left[ind],
                      loc_diff=self._loc_diff[ind])
            label(self._top_left[ind], text_list[ind] + '\n{}@{}x{}'.format(
                self.conv_layers_n[ind], self.conv_layers_sizes[ind],
                self.conv_layers_sizes[ind]))

    def add_mappings(self):
        # start_ratio_list = [[0.4, 0.5], [0.4, 0.8], [0.4, 0.5], [0.4, 0.8]]
        start_ratio_list = [[0.4, 0.5]] * len(self.mappings_sizes)
        # ind_bgn_list = range(len(self.mappings_sizes))
        text_list = ['Convolution', 'Max-pooling', 'Convolution',
                     'Max-pooling']

        for ind in range(len(self.mappings_sizes)):
            add_mapping(self._patches, self._lines, self._colors,
                        start_ratio_list[ind],
                        self.mappings_sizes[ind], ind,
                        self._top_left, self._loc_diff, self._num_show,
                        self.conv_layers_sizes)
            label(
                self._top_left[ind], text_list[ind] + '\n{}x{} kernel'.format(
                    self.mappings_sizes[ind], self.mappings_sizes[ind]),
                xy_off=[26, -65])
