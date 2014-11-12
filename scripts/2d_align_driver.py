# A driver for running 2D alignment using the FijiBento alignment project
# The input is a directory that contains image files (tiles), and the output is a 2D montage of these files
# Activates ComputeSIFTFeaturs -> MatchSIFTFeatures -> OptimizeMontageTransfrom
# and the result can then be rendered if needed
#
# requires:
# - java (executed from the command line)
# - 

import sys
import os
import argparse
import json

from filter_tiles import filter_tiles
from create_sift_features import create_sift_features
from match_sift_features import match_sift_features
from match_sift_features_and_filter import match_sift_features_and_filter
from json_concat import json_concat
from optimize_montage_transform import optimize_montage_transform
from match_by_max_pmcc import match_by_max_pmcc
from optimize_elastic_transform import optimize_elastic_transform


# Command line parser
parser = argparse.ArgumentParser(description='A driver that does a 2D alignment of images.')
parser.add_argument('tiles_fname', metavar='tiles_json', type=str, 
                    help='a tile_spec file that contains all the images to be aligned in json format')
parser.add_argument('-w', '--workspace_dir', type=str, 
                    help='a directory where the output files of the different stages will be kept (default: current directory)',
                    default='.')
parser.add_argument('-r', '--render', action='store_true',
                    help='render final result')
parser.add_argument('-o', '--output_file_name', type=str, 
                    help='the file that includes the output to be rendered in json format (default: output.json)',
                    default='output.json')
parser.add_argument('-f', '--fixed_tiles', type=str, nargs='+',
                    help='a space separated list of fixed tile indices (default: 0)',
                    default="0")
parser.add_argument('-j', '--jar_file', type=str, 
                    help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                    default='../target/render-0.0.1-SNAPSHOT.jar')
# the default bounding box is as big as the image can be
parser.add_argument('-b', '--bounding_box', type=str, 
                    help='the bounding box of the part of image that needs to be aligned format: "from_x to_x from_y to_y" (default: all tiles)',
                    default='{0} {1} {2} {3}'.format((-sys.maxint - 1), sys.maxint, (-sys.maxint - 1), sys.maxint))
parser.add_argument('-c', '--conf_file_name', type=str, 
                    help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if )',
                    default=None)



args = parser.parse_args()

print args

# create a workspace directory if not found
if not os.path.exists(args.workspace_dir):
    os.makedirs(args.workspace_dir)

conf = None
if not args.conf_file_name is None:
    with open(args.conf_file_name, 'r') as conf_file:
        conf = json.load(conf_file)

tiles_fname_prefix = os.path.splitext(os.path.basename(args.tiles_fname))[0]

# filter the tiles to the requested bounding box
filter_json = os.path.join(args.workspace_dir, "{0}_filterd.json".format(tiles_fname_prefix))
filter_tiles(args.tiles_fname, filter_json, args.bounding_box)


# create the sift features of these tiles
sifts_json = os.path.join(args.workspace_dir, "{0}_sifts.json".format(tiles_fname_prefix))
##create_sift_features(filter_json, sifts_json, args.jar_file, conf)

# match the features of overlapping tiles
match_json = os.path.join(args.workspace_dir, "{0}_sift_matches.json".format(tiles_fname_prefix))
#match_sift_features(filter_json, sifts_json, match_json, args.jar_file, conf)
##match_sift_features_and_filter(filter_json, sifts_json, match_json, args.jar_file, conf)

# optimize the 2d layer montage
optmon_fname = os.path.join(args.workspace_dir, "{0}_optimized_montage.json".format(tiles_fname_prefix))
##optimize_montage_transform(match_json, filter_json, args.fixed_tiles, optmon_fname, args.jar_file, conf)

# template matching by max PMCC
pmcc_json = os.path.join(args.workspace_dir, "{0}_pmcc_matches.json".format(tiles_fname_prefix))
##match_by_max_pmcc(optmon_fname, args.fixed_tiles, pmcc_json, args.jar_file, conf)

# optimize the 2d layer elastically
optmon_elastic_fname = os.path.join(args.workspace_dir, "{0}_opt_elastic_montage.json".format(tiles_fname_prefix))
optimize_elastic_transform(pmcc_json, filter_json, args.fixed_tiles, optmon_elastic_fname, args.jar_file, conf)



