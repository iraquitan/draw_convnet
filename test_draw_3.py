from dnn_draw.net import ConvNet

bn = ConvNet('my_cnn', 60)

# Input layer
bn.add_input(1, 32)

# Conv layers
bn.add_conv_layer('Convolution', 32, 5, padding='valid',
                  start_ratio=[0.4, 0.5])
# bn.add_conv_layer('Max-pooling', 32, 2, start_ratio=[0.4, 0.8])
bn.add_conv_layer('Convolution', 32, 5, padding='valid',
                  start_ratio=[0.4, 0.5])
bn.add_conv_layer('Max-pooling', 32, 2, type='pool', padding='valid',
                  stride=2, start_ratio=[0.4, 0.8])

# Fully connected layers
bn.add_fc_layer(128)
bn.add_fc_layer(128)
bn.add_fc_layer(64)

# Output layer
bn.add_output(4)

bn.draw(False)
bn.save_plot()
