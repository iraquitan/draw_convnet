from draw_convnet.draw import ConvNet

cn_sizes = [32, 18, 10, 6, 4]
cn_ns = [3, 32, 32, 48, 48]
map_sizes = [5, 2, 5, 2]
map_labels = ['Convolution', 'Max-pooling', 'Convolution', 'Max-pooling']
map_st_ratio = [[0.4, 0.5], [0.4, 0.8], [0.4, 0.5], [0.4, 0.8]]
fc_sizes = [2, 2, 2]
fc_labels = ['Flatten\n', 'Fully\nconnected', 'Fully\nconnected']
fc_ns = [768, 500, 2]

cn = ConvNet(cn_sizes, cn_ns, map_sizes, map_labels, map_st_ratio, fc_sizes,
             fc_labels, fc_ns)

cn.add_conv_layers()
cn.add_mappings()
cn.add_fc_layers()
cn.plot()
