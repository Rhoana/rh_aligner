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
import itertools
from bounding_box import BoundingBox

from filter_tiles import filter_tiles
from create_sift_features_cv2 import create_sift_features
from create_surf_features_cv2 import create_surf_features
#from match_sift_features import match_sift_features
from match_sift_features_and_filter_cv2 import match_single_sift_features_and_filter
from json_concat import json_concat
from optimize_montage_transform import optimize_montage_transform
from utils import write_list_to_file


def load_tilespecs(tile_file):
    tile_file = tile_file.replace('file://', '')
    with open(tile_file, 'r') as data_file:
        tilespecs = json.load(data_file)

    return tilespecs

# Command line parser
parser = argparse.ArgumentParser(description='A driver that does a 2D affine alignment of images.')
parser.add_argument('tiles_fname', metavar='tiles_json', type=str, 
                    help='a tile_spec file that contains all the images to be aligned in json format')
parser.add_argument('-w', '--workspace_dir', type=str, 
                    help='a directory where the output files of the different stages will be kept (default: current directory)',
                    default='.')
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
parser.add_argument('-t', '--threads_num', type=int, 
                    help='the number of threads to use (default: 1)',
                    default=None)




args = parser.parse_args()

print args

# create a workspace directory if not found
if not os.path.exists(args.workspace_dir):
    os.makedirs(args.workspace_dir)

tiles_fname_prefix = os.path.splitext(os.path.basename(args.tiles_fname))[0]


# read tile spec and find the features for each tile
tilespecs = load_tilespecs(args.tiles_fname)
all_features = {}
all_matched_features = []

for i, ts in enumerate(tilespecs):
    imgurl = ts["mipmapLevels"]["0"]["imageUrl"]
    tile_fname = os.path.basename(imgurl).split('.')[0]

    # create the features of these tiles
    features_json = os.path.join(args.workspace_dir, "{0}_sifts_{1}.hdf5".format(tiles_fname_prefix, tile_fname))
    if not os.path.exists(features_json):
        create_sift_features(args.tiles_fname, features_json, i, args.conf_file_name)
    all_features[imgurl] = features_json


# read every pair of overlapping tiles, and match their sift features

# TODO: add all tiles to a kd-tree so it will be faster to find overlap between tiles

# iterate over the tiles, and for each tile, find intersecting tiles that overlap,
# and match their features
# Nested loop:
#    for each tile_i in range[0..N):
#        for each tile_j in range[tile_i..N)]
indices = []
for pair in itertools.combinations(xrange(len(tilespecs)), 2):
    idx1 = pair[0]
    idx2 = pair[1]
    ts1 = tilespecs[idx1]
    ts2 = tilespecs[idx2]
    # if the two tiles intersect, match them
    bbox1 = BoundingBox.fromList(ts1["bbox"])
    bbox2 = BoundingBox.fromList(ts2["bbox"])
    if bbox1.overlap(bbox2):
        imageUrl1 = ts1["mipmapLevels"]["0"]["imageUrl"]
        imageUrl2 = ts2["mipmapLevels"]["0"]["imageUrl"]
        tile_fname1 = os.path.basename(imageUrl1).split('.')[0]
        tile_fname2 = os.path.basename(imageUrl2).split('.')[0]
        print "Matching features of tiles: {0} and {1}".format(imageUrl1, imageUrl2)
        index_pair = [idx1, idx2]
        match_json = os.path.join(args.workspace_dir, "{0}_sift_matches_{1}_{2}.json".format(tiles_fname_prefix, tile_fname1, tile_fname2))
        # match the features of overlapping tiles
        if not os.path.exists(match_json):
            match_single_sift_features_and_filter(args.tiles_fname, all_features[imageUrl1], all_features[imageUrl2], match_json, index_pair, conf_fname=args.conf_file_name)
        all_matched_features.append(match_json)

# Create a single file that lists all tilespecs and a single file that lists all pmcc matches (the os doesn't support a very long list)
matches_list_file = os.path.join(args.workspace_dir, "all_matched_sifts_files.txt")
write_list_to_file(matches_list_file, all_matched_features)

# optimize the 2d layer montage
optmon_fname = os.path.join(args.workspace_dir, "{0}_optimized_montage.json".format(tiles_fname_prefix))
if not os.path.exists(optmon_fname):
    optimize_montage_transform(matches_list_file, args.tiles_fname, args.fixed_tiles, optmon_fname, args.jar_file, args.conf_file_name, args.threads_num)



