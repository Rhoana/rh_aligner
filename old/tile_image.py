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
import cv2
import multiprocessing as mp


def create_single_tile(img, out_file, bbox):
    print "Saving tile using bbox: {} to: {}".format(bbox, out_file)
    img_crop = img[bbox[2]:bbox[3], bbox[0]:bbox[1]]
    cv2.imwrite(out_file, img_crop)

def create_tiles(img, out_file_bbox_list):
    for out_file, bbox in out_file_bbox_list:
        create_single_tile(img, out_file, bbox)

def split(lst, n):
    res = []
    prev_start_idx = 0
    for i in range(n):
        next_idx = prev_start_idx + len(lst)//n
        if i < len(lst) % n:
            next_idx += 1
        res.append(lst[prev_start_idx:next_idx])
        prev_start_idx = next_idx
    return res
    #num_extra = len(iterable) % n
    #zipped = zip(*[iter(iterable)] * n)
    #return zipped if not num_extra else zipped + [iterable[-num_extra:], ]

def tile_image(img_file, output_dir, processes_num=1, tile_size=512, output_type='png', output_pattern=None):

    if output_pattern is None:
        output_pattern = "{}%rowcol".format(os.path.splitext(os.path.basename(img_file))[-1])

    # Read image file into an array
    print "Reading original image file size"
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

    # Find the x and y tick marks
    xs = np.arange(0, img.shape[1], tile_size)
    ys = np.arange(0, img.shape[0], tile_size)

    # Create the task list to distribute between the processes
    tasks_list = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            bbox = [x, min(x + tile_size, img.shape[1]), y, min(y + tile_size, img.shape[0])]
            file_name = "{}.{}".format(output_pattern.replace('%rowcol', '_tr{}-tc{}_'.format(j + 1, i + 1)), output_type)
            out_file = os.path.join(output_dir, file_name)
            tasks_list.append([out_file, bbox])

    print "Starting tiles creation, with {} tasks (tiles)".format(len(tasks_list))

    if processes_num == 1: # Single process execution
        create_tiles(img, tasks_list)
    else: # Multiple processes
        
        # create N-1 worker processes
        pool = mp.Pool(processes=processes_num - 1)
        print "Creating {} other processes".format(processes_num - 1)

        # Each processes will receive ~ len(tasks_list)//processes_num tasks
        per_worker_tasks = split(tasks_list, processes_num)

        for i in range(processes_num - 1):
            async_res = pool.apply_async(create_tiles, (img, per_worker_tasks[i]))

        # Get the current process to do the rest of the work
        create_tiles(img, per_worker_tasks[-1])

        # wait for all other processes to finish their job
        if pool is not None:
            pool.close()
            pool.join()
    print "Done"


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Receives a single image file, and creates tiles from that.')
    parser.add_argument('img_file', metavar='img_file', type=str, 
                        help='the image file to tile')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='The directory where the tiled images will be at (default: .)',
                        default='./')
    parser.add_argument('-p', '--processes_num', type=int, 
                        help='Number of python processes to create the tiles (default: 1)',
                        default=1)
    parser.add_argument('-s', '--tile_size', type=int,
                        help='the size (square side) of each tile (default: 512)',
                        default=512)
    parser.add_argument('--output_type', type=str,
                        help='The output type format (default: png)',
                        default='png')
    parser.add_argument('--output_pattern', type=str,
                        help='The output file name pattern where "%rowcol" will be replaced by "_tr[row]-tc[rol]_" with the row and column numbers',
                        default=None)

    args = parser.parse_args()

    #print args
    utils.create_dir(args.output_dir)
    tile_image(args.img_file, args.output_dir, args.processes_num, args.tile_size, args.output_type, args.output_pattern)

if __name__ == '__main__':
    main()

