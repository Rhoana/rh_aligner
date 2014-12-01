# Receives a directory which has the tiled images of some section full resultion,
# and creates zoomed downsampled tiles (mipmaps), up to a level which has the entire image size as a single original tile
#
# Note that all tiles are assumed to be of the same size (and tile width == tile height),
# and that the given directory includes a sub-directory named "0" that has the full resultion tiles
# 

import sys
import os
import glob
import argparse
import utils
import math
import numpy as np
from scipy import ndimage
import matplotlib.image as mpimg
import cv2


def block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx/fact, sy/fact)
    return res

def get_tile_size(tile_file):
    img = mpimg.imread(tile_file)
    return img.shape[0]

def create_img(tiles_dir, from_row, from_col, rows_num, cols_num, tile_size):
    new_img_shape = (tile_size * rows_num, tile_size * cols_num)
    new_img_arr = np.zeros(new_img_shape, dtype='uint8')
    for row in range(rows_num):
        for col in range(cols_num):

            tile_files = glob.glob(os.path.join(tiles_dir, 'tile_{}_{}.*'.format(from_row + row, from_col + col)))
            if len(tile_files) > 0:
                tile_file = tile_files[0]
                tile_img = mpimg.imread(tile_file)
                new_img_arr[row * tile_size:(row + 1) * tile_size,
                            col * tile_size:(col + 1) * tile_size] = tile_img
    return new_img_arr

def create_zoomed_tiles(tiles_dir):
    #all_tiles = glob.glob(os.path.join(os.path.join(tiles_dir, '0'), '*'))

    first_tile = glob.glob(os.path.join(os.path.join(tiles_dir, '0'), 'tile_0_0.*'))[0]
    tile_ext = os.path.splitext(first_tile)[1]
    tile_size = get_tile_size(first_tile)

    # get number of rows and columns used for full resolution
    cur_rows = len(glob.glob(os.path.join(os.path.join(tiles_dir, '0'), 'tile_*_0.*')))
    cur_cols = len(glob.glob(os.path.join(os.path.join(tiles_dir, '0'), 'tile_0_*')))

    single_tile = (cur_rows == 1) and (cur_cols == 1)
    zoom_level = 0

    while not single_tile:
        prev_dir = os.path.join(tiles_dir, '{}'.format(zoom_level))
        zoom_level += 1
        cur_dir = os.path.join(tiles_dir, '{}'.format(zoom_level))
        utils.create_dir(cur_dir)
        prev_rows = cur_rows
        prev_cols = cur_cols
        cur_rows =  int(math.ceil(float(prev_rows) / 2))
        cur_cols =  int(math.ceil(float(prev_cols) / 2))
        # create the downscaled images and save them
        for row in range(cur_rows):
            for col in range(cur_cols):
                # load the original 4 tiles image
                new_img = create_img(prev_dir, row * 2, col * 2, 2, 2, tile_size)
                # out_file_new = os.path.join(cur_dir, 'tile_{}_{}_new{}'.format(row, col, tile_ext))
                # mpimg.imsave(out_file_new, new_img)
                # scale down the image by 2 and save it to disk
                scaled_img = block_mean(new_img, 2)
                out_file = os.path.join(cur_dir, 'tile_{}_{}{}'.format(row, col, tile_ext))
                cv2.imwrite(out_file, scaled_img)
        single_tile = (cur_rows == 1) and (cur_cols == 1)



def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Receives a directory which has the tiled images of some section full resultion, \
                        and creates zoomed downsampled tiles (mipmaps), up to a level which has the entire image size as a single original tile.')
    parser.add_argument('tiles_dir', metavar='tiles_dir', type=str, 
                        help='the directory which has a sub-directory named "0" that contains the full resolution tiles')


    args = parser.parse_args()

    #print args

    create_zoomed_tiles(args.tiles_dir)
    # try:
    #     create_zoomed_tiles(args.tiles_dir)
    # except:
    #     sys.exit("Error while executing: {0}".format(sys.argv))

if __name__ == '__main__':
    main()

