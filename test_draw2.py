from dnn_draw.net import BaseNet

bn = BaseNet('my_cnn')

# Conv layers
bn.add_conv_layer('Inputs', 32, 3)
bn.add_conv_layer('Feature\nmaps', 18, 32)
bn.add_conv_layer('Feature\nmaps', 10, 32)
bn.add_conv_layer('Feature\nmaps', 6, 48)
bn.add_conv_layer('Feature\nmaps', 4, 48)

# Fully connected layers
bn.add_fc_layer('Hidden\nunits', 768)
bn.add_fc_layer('Hidden\nunits', 500)
bn.add_fc_layer('Outputs', 2)

# Mappings
bn.add_conv_mapping('Convolution', 0, 1, 5, [0.4, 0.5])
bn.add_conv_mapping('Max-pooling', 1, 2, 2, [0.4, 0.8])
bn.add_conv_mapping('Convolution', 2, 3, 5, [0.4, 0.5])
bn.add_conv_mapping('Max-pooling', 3, 4, 2, [0.4, 0.8])

bn.draw(False)
bn.save_plot()
