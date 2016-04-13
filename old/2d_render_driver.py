# A driver for rendering 2D images using the FijiBento alignment project
# The input is a directory that contains image files (tilespecs) where each file is of a single section,
# and the output is a 2D montage of these sections
#
# requires:
# - java (executed from the command line)
# - 

import sys
import os
import argparse
import json
import utils

from render_2d import render_2d
from normalize_coordinates import normalize_coordinates


# Command line parser
parser = argparse.ArgumentParser(description='A driver that does a 2D rendering of tilespec images.')
parser.add_argument('tiles_fname', metavar='tiles_fname', type=str, 
                    help='a tile_spec (json) file that contains a single section to be rendered')
parser.add_argument('-w', '--workspace_dir', type=str, 
                    help='a directory where the output files of the different stages will be kept (default: ./2d_render_workdir)',
                    default='./2d_render_workdir')
parser.add_argument('-o', '--output_fname', type=str, 
                    help='the output file (default: ./[tiles_fname].tif)',
                    default=None)
parser.add_argument('-j', '--jar_file', type=str, 
                    help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                    default='../target/render-0.0.1-SNAPSHOT.jar')
parser.add_argument('-t', '--threads_num', type=int, 
                    help='the number of threads to use (default: number of cores in the system)',
                    default=None)




args = parser.parse_args()

print args

utils.create_dir(args.workspace_dir)
norm_dir = os.path.join(args.workspace_dir, "normalized")
utils.create_dir(norm_dir)

tiles_fname_basename = os.path.basename(args.tiles_fname)
tiles_fname_prefix = os.path.splitext(tiles_fname_basename)[0]

# Normalize the json file
norm_json = os.path.join(norm_dir, tiles_fname_basename)
if not os.path.exists(norm_json):
    normalize_coordinates(args.tiles_fname, norm_dir, args.jar_file)

# Render the normalized json file
out_fname = args.output_fname
if not os.path.exists(out_fname):
    render_2d(norm_json, out_fname, -1, args.jar_file, args.threads_num)

