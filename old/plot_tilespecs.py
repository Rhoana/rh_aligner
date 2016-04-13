# Plot a given tilespecs file (mfov centers and mfov boundaries)
import sys
import os
import matplotlib
matplotlib.use('GTK')
import matplotlib.pyplot as plt
#import matplotlib
import json
import numpy as np
import utils

def get_center(mfov_ts):
    xlocsum, ylocsum, nump = 0, 0, 0
    for tile_ts in mfov_ts.values():
        xlocsum += tile_ts["bbox"][0] + tile_ts["bbox"][1]
        ylocsum += tile_ts["bbox"][2] + tile_ts["bbox"][3]
        nump += 2
    return [xlocsum / nump, ylocsum / nump]


def plot_tilespecs(ts_file):
    # Read the tilespecs file
    ts = utils.load_tilespecs(ts_file)
    # Index the tilespecs according to mfov and tile_index (1-based)
    indexed_ts = utils.index_tilespec(ts)

    # Get the centers
    centers = [get_center(indexed_ts[m]) for m in sorted(indexed_ts.keys())]
    max_x_or_y = np.max(centers)

    # Create the figure
    fig = plt.figure()
    fig.suptitle('{} - mfovs'.format(os.path.basename(ts_file)), fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)


    # plot text at the centers location
    for i, center in enumerate(centers):
        print (i+1), center
        center_x = int(center[0] / max_x_or_y * 1000)
        center_y = int(center[1] / max_x_or_y * 1000)
        ax.text(center_x, center_y, str(i + 1), fontsize=15)

    ax.axis([0, 1000, 0, 1000])
        
    # TODO - plot the boundaries of the mfovs

    # plot the entire graph
    plt.show()

    return fig
    

if __name__ == '__main__':
    ts_file = sys.argv[1]

    img = plot_tilespecs(ts_file)
    # plot the image


