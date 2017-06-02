import matplotlib as mpl
try:
    import matplotlib.pyplot as plt
except RuntimeError as e:
    mpl.rcParams['backend'] = 'TkAgg'


def label(xy, text, xy_off=[0, 4], axes=None):
    if axes is None:
        plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
                 family='sans-serif', size=8)
    else:
        axes.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
                  family='sans-serif', size=8)
