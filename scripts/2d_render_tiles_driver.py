# A driver for rendering 2D images using the FijiBento alignment project
# The input is a tilespec (json) file of a single section,
# and the output is a directory with squared tiles of the 2D montage of the sections
#
# requires:
# - java (executed from the command line)
# - 

import sys
import os
import argparse
import json
import utils

from render_tiles_2d import render_tiles_2d
from normalize_coordinates import normalize_coordinates
from create_zoomed_tiles import create_zoomed_tiles


# Command line parser
parser = argparse.ArgumentParser(description='A driver that does a 2D rendering of tilespec images.')
parser.add_argument('tiles_fname', metavar='tiles_fname', type=str, 
                    help='a tile_spec (json) file that contains a single section to be rendered')
parser.add_argument('-w', '--workspace_dir', type=str, 
                    help='a directory where the output files of the different stages will be kept (default: ./2d_render_workdir)',
                    default='./2d_render_workdir')
parser.add_argument('-o', '--output_dir', type=str, 
                    help='the output directory (default: ./output_tiles)',
                    default='./output_tiles')
parser.add_argument('-j', '--jar_file', type=str, 
                    help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                    default='../target/render-0.0.1-SNAPSHOT.jar')
parser.add_argument('-t', '--threads_num', type=int, 
                    help='the number of threads to use (default: number of cores in the system)',
                    default=None)
parser.add_argument('-s', '--tile_size', type=int, 
                    help='the size (square side) of each tile (default: 512)',
                    default=512)



args = parser.parse_args()

print args

utils.create_dir(args.workspace_dir)
norm_dir = os.path.join(args.workspace_dir, "normalized")
utils.create_dir(norm_dir)

utils.create_dir(args.output_dir)

tiles_fname_basename = os.path.basename(args.tiles_fname)
tiles_fname_prefix = os.path.splitext(tiles_fname_basename)[0]

# Normalize the json file
norm_json = os.path.join(norm_dir, tiles_fname_basename)
if not os.path.exists(norm_json):
    normalize_coordinates(args.tiles_fname, norm_dir, args.jar_file)

# Render the normalized json file
out_0_dir = os.path.join(args.output_dir, "0")
if not os.path.exists(out_0_dir):
    render_tiles_2d(norm_json, out_0_dir, args.tile_size, args.jar_file, args.threads_num)

# create the zoomed tiles
out_1_dir = os.path.join(args.output_dir, "1")
if not os.path.exists(out_1_dir):
    create_zoomed_tiles(args.output_dir, True, args.threads_num)
