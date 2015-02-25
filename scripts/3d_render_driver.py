# A driver for running 3D alignment using the FijiBento alignment project
# The input is two tile spec files with their 2d alignment
# and each file has also a z axis (layer) index, and the output is a tile spec after 3D alignment
# Activates ComputeSIFTFeaturs -> MatchSIFTFeatures -> OptimizeSeriesTransfrom -> MatchByMaxPMCC -> OptimizeSeriesElastic
# and the result can then be rendered if needed
#
# requires:
# - java (executed from the command line)
# - 

import sys
import os
import argparse
import json
import glob

from update_bounding_box import update_bounding_box
from normalize_coordinates import normalize_coordinates
from render_3d import render_3d
from utils import path2url, write_list_to_file, create_dir, read_layer_from_file, parse_range
from bounding_box import BoundingBox




# Command line parser
parser = argparse.ArgumentParser(description='A driver that does a 3D alignment of images.')
parser.add_argument('input_dir', metavar='input_dir', type=str, 
                    help='a directory that contains all the tile_spec files of all sections to be rendered (each section in a single tile_spec file) json format')
parser.add_argument('-w', '--workspace_dir', type=str, 
                    help='a directory where the output files of the different stages will be kept (default: ./work_dir)',
                    default='./work_dir')
parser.add_argument('-o', '--output_dir', type=str, 
                    help='the directory where the output rendered files will be stored (default: ./output)',
                    default='./output')
parser.add_argument('-j', '--jar_file', type=str, 
                    help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                    default='../target/render-0.0.1-SNAPSHOT.jar')
parser.add_argument('--from_layer', type=int, 
                    help='the layer to start from (inclusive, default: the first layer in the data)',
                    default=-1)
parser.add_argument('--to_layer', type=int, 
                    help='the last layer to render (inclusive, default: the last layer in the data)',
                    default=-1)
parser.add_argument('--width', type=int, 
                    help='set the width of the rendered images (default: full image)',
                    default=-1)
parser.add_argument('-s', '--skip_layers', type=str, 
                    help='the range of layers (sections) that will not be processed e.g., "2,3,9-11,18" (default: no skipped sections)',
                    default=None)
parser.add_argument('-t', '--threads', type=int, 
                    help='the number of threads to use (default: 4)',
                    default=4)
parser.add_argument('--skip_bounding_box', action="store_true",
                    help='skip the phase that computes the new bounding box (default: no)')



args = parser.parse_args()

print args

# create a workspace directory if not found
create_dir(args.workspace_dir)


bbox_dir = os.path.join(args.workspace_dir, "after_bbox")
create_dir(bbox_dir)
norm_dir = os.path.join(args.workspace_dir, "after_norm")
create_dir(norm_dir)

create_dir(args.output_dir)

all_layers = []
layer_to_bbox = {}
layer_to_ts_json = {}
layer_to_json_prefix = {}
layer_meshes_dir = {}
all_norm_files = []

skipped_layers = parse_range(args.skip_layers)


for tiles_fname in glob.glob(os.path.join(args.input_dir, '*.json')):
    tiles_fname_prefix = os.path.splitext(os.path.basename(tiles_fname))[0]

    # read the layer from the file
    layer = read_layer_from_file(tiles_fname)

    if args.from_layer != -1:
        if layer < args.from_layer:
            continue
    if args.to_layer != -1:
        if layer > args.to_layer:
            continue
    if layer in skipped_layers:
        continue


    all_layers.append(layer)

    # Update the bounding box of the section
    if args.skip_bounding_box:
        bbox_json = tiles_fname
    else:
        print "Updating bounding box of {0}".format(tiles_fname_prefix)
        bbox_json = os.path.join(bbox_dir, "{0}.json".format(tiles_fname_prefix))
        if not os.path.exists(bbox_json):
            update_bounding_box(tiles_fname, bbox_dir, args.jar_file, threads_num=args.threads)
    layer_to_bbox[layer] = bbox_json
    layer_to_json_prefix[layer] = tiles_fname_prefix
    layer_to_ts_json[layer] = tiles_fname
    all_norm_files.append(os.path.join(norm_dir, "{}.json".format(tiles_fname_prefix)))



# Verify that all the layers are there and that there are no holes
all_layers.sort()
for i in range(len(all_layers) - 1):
    if all_layers[i + 1] - all_layers[i] != 1:
        for l in range(all_layers[i] + 1, all_layers[i + 1]):
            if l not in skipped_layers:
                print "Error missing layer {} between: {} and {}".format(l, all_layers[i], all_layers[i + 1])
                sys.exit(1)

# Normalize the sections
print [layer_to_bbox[l] for l in layer_to_bbox.keys()]
normalize_coordinates([layer_to_bbox[l] for l in layer_to_bbox.keys()], norm_dir, args.jar_file)

norm_list_file = os.path.join(args.workspace_dir, "all_norm_files.txt")
write_list_to_file(norm_list_file, all_norm_files)



# Render each layer individually
for tiles_fname in glob.glob(os.path.join(norm_dir, '*.json')):
    tiles_fname_prefix = os.path.splitext(os.path.basename(tiles_fname))[0]

    # read the layer from the file
    layer = read_layer_from_file(tiles_fname)


    # Check if it already rendered the files (don't know the output type)
    render_out_files = glob.glob(os.path.join(args.output_dir, '{0:0>4}_{1}.*'.format(layer, tiles_fname_prefix)))
    if len(render_out_files) > 0:
        print "Skipping rendering of layer {}, because found: {}".format(layer, render_out_files)
        continue


    print "Rendering {0}".format(tiles_fname_prefix)
    render_3d(norm_list_file, args.output_dir, layer, layer, args.width, args.jar_file, threads_num=args.threads)


