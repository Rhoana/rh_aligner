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

#from filter_tiles import filter_tiles
from update_bbox import update_bbox, read_bbox
#from create_sift_features import create_sift_features
from create_layer_sift_features import create_layer_sift_features
from match_layers_sift_features import match_layers_sift_features
from filter_ransac import filter_ransac
from match_layers_by_max_pmcc import match_layers_by_max_pmcc
from optimize_layers_elastic import optimize_layers_elastic
from utils import path2url


def read_layer_from_file(tiles_spec_fname):
    layer = None
    with open(tiles_spec_fname, 'r') as data_file:
        data = json.load(data_file)
    for tile in data:
        if tile['layer'] is None:
            print "Error reading layer in one of the tiles in: {0}".format(tiles_spec_fname)
            sys.exit(1)
        if layer is None:
            layer = tile['layer']
        if layer != tile['layer']:
            print "Error when reading tiles from {0} found inconsistent layers numbers: {1} and {2}".format(tiles_spec_fname, layer, tile['layer'])
            sys.exit(1)
    if layer is None:
        print "Error reading layers file: {0}. No layers found.".format(tiles_spec_fname)
        sys.exit(1)
    return int(layer)


def create_dir(path):
    # create a directory if not found
    if not os.path.exists(path):
        os.makedirs(path)


# Command line parser
parser = argparse.ArgumentParser(description='A driver that does a 3D alignment of images.')
parser.add_argument('input_dir', metavar='input_dir', type=str, 
                    help='a directory that contains all the tile_spec files of all sections (each section already aligned and in a single tile_spec file) json format')
parser.add_argument('-w', '--workspace_dir', type=str, 
                    help='a directory where the output files of the different stages will be kept (default: ./work_dir)',
                    default='./work_dir')
parser.add_argument('-r', '--render', action='store_true',
                    help='render final result')
parser.add_argument('-o', '--output_dir', type=str, 
                    help='the directory where the output to be rendered in json format files will be stored (default: ./output)',
                    default='./output')
parser.add_argument('-j', '--jar_file', type=str, 
                    help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                    default='../target/render-0.0.1-SNAPSHOT.jar')
# the default bounding box is as big as the image can be
parser.add_argument('-b', '--bounding_box', type=str, 
                    help='the bounding box of the part of image that needs to be aligned format: "from_x to_x from_y to_y" (default: all tiles)',
                    default='{0} {1} {2} {3}'.format((-sys.maxint - 1), sys.maxint, (-sys.maxint - 1), sys.maxint))
parser.add_argument('-d', '--max_layer_distance', type=int, 
                    help='the largest distance between two layers to be matched (default: 1)',
                    default=1)
parser.add_argument('-c', '--conf_file_name', type=str, 
                    help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if )',
                    default=None)



args = parser.parse_args()

print args

# create a workspace directory if not found
create_dir(args.workspace_dir)

conf = None
#if not args.conf_file_name is None:
#    with open(args.conf_file_name, 'r') as conf_file:
#        conf = json.load(conf_file)

after_bbox_dir = os.path.join(args.workspace_dir, "after_bbox")
create_dir(after_bbox_dir)
sifts_dir = os.path.join(args.workspace_dir, "sifts")
create_dir(sifts_dir)
matched_sifts_dir = os.path.join(args.workspace_dir, "matched_sifts")
create_dir(matched_sifts_dir)
after_ransac_dir = os.path.join(args.workspace_dir, "after_ransac")
create_dir(after_ransac_dir)
matched_pmcc_dir = os.path.join(args.workspace_dir, "matched_pmcc")
create_dir(matched_pmcc_dir)
after_elastic_dir = os.path.join(args.workspace_dir, "after_elastic")
create_dir(after_elastic_dir)

all_layers = []
layer_to_sifts = {}
layer_to_ts_json = {}
layer_to_json_prefix = {}

# Find all images width and height (for the mesh)
imageWidth = None
imageHeight = None

bbox_suffix = "_bbox"

for tiles_fname in glob.glob(os.path.join(args.input_dir, '*.json')):
    tiles_fname_prefix = os.path.splitext(os.path.basename(tiles_fname))[0]

    # read the layer from the file
    layer = read_layer_from_file(tiles_fname)

    all_layers.append(layer)

    # update the bbox of each section
    #after_bbox_json = os.path.join(after_bbox_dir, "{0}{1}.json".format(tiles_fname_prefix, bbox_suffix)) 
    #if not os.path.exists(after_bbox_json):
    #    print "Updating bounding box of {0}".format(tiles_fname_prefix)
    #    update_bbox(args.jar_file, tiles_fname, out_dir=after_bbox_dir, out_suffix=bbox_suffix)
    #bbox = read_bbox(after_bbox_json)
    bbox = read_bbox(tiles_fname)
    if imageWidth is None or imageWidth < (bbox[1] - bbox[0]):
        imageWidth = bbox[1] - bbox[0]
    if imageHeight is None or imageHeight < bbox[3] - bbox[2]:
        imageHeight = bbox[3] - bbox[2]

    # create the sift features of these tiles
    print "Computing sift features of {0}".format(tiles_fname_prefix)
    sifts_json = os.path.join(sifts_dir, "{0}_sifts.json".format(tiles_fname_prefix))
    if not os.path.exists(sifts_json):
        #create_layer_sift_features(after_bbox_json, sifts_json, args.jar_file, conf)
        create_layer_sift_features(tiles_fname, sifts_json, args.jar_file, conf)
    layer_to_sifts[layer] = sifts_json
    layer_to_json_prefix[layer] = tiles_fname_prefix
    #layer_to_ts_json[layer] = after_bbox_json
    layer_to_ts_json[layer] = tiles_fname

# Verify that all the layers are there and that there are no holes
all_layers.sort()
for i in range(len(all_layers) - 1):
    if all_layers[i + 1] - all_layers[i] != 1:
        print "Error missing layers between: {1} and {2}".format(all_layers[i], all_layers[i + 1])
        sys.exit(1)

print "Found the following layers: {0}".format(all_layers)

print "All json files prefix are: {0}".format(layer_to_json_prefix)

fixed_layer = all_layers[0]


# Match and optimize each two layers in the required distance
all_pmcc_files = []
for i in all_layers:
    layers_to_process = min(i + args.max_layer_distance, all_layers[-1] + 1) - i
    print "layers_to_process {0}".format(layers_to_process)
    for j in range(1, layers_to_process):
        print "j {0}".format(j)
        fname1_prefix = layer_to_json_prefix[i]
        fname2_prefix = layer_to_json_prefix[i + j]

        # match the features of neighboring tiles
        match_json = os.path.join(matched_sifts_dir, "{0}_{1}_sift_matches.json".format(fname1_prefix, fname2_prefix))
        if not os.path.exists(match_json):
            print "Matching layers' sifts: {0} and {1}".format(i, i + j)
            match_layers_sift_features(layer_to_ts_json[i], layer_to_sifts[i], \
                layer_to_ts_json[i + j], layer_to_sifts[i + j], match_json, args.jar_file, conf)

        # filter and ransac the matched points
        ransac_fname = os.path.join(after_ransac_dir, "{0}_{1}_filter_ransac.json".format(fname1_prefix, fname2_prefix))
        if not os.path.exists(ransac_fname):
            print "Filter-and-Ransac of layers: {0} and {1}".format(i, i + j)
            filter_ransac(match_json, path2url(layer_to_ts_json[i]), ransac_fname, args.jar_file, conf)

        # match by max PMCC the two layers
        pmcc_fname = os.path.join(matched_pmcc_dir, "{0}_{1}_match_pmcc.json".format(fname1_prefix, fname2_prefix))
        if not os.path.exists(pmcc_fname):
            print "Matching layers by Max PMCC: {0} and {1}".format(i, i + j)
            match_layers_by_max_pmcc(args.jar_file, layer_to_ts_json[i], layer_to_ts_json[i + j], ransac_fname, imageWidth, imageHeight, [fixed_layer], pmcc_fname, conf)
        all_pmcc_files.append(pmcc_fname)

print "All pmcc files: {0}".format(all_pmcc_files)

# Optimize all layers to a single 3d image
all_ts_files = layer_to_ts_json.values()
create_dir(args.output_dir)
optimize_layers_elastic(all_ts_files, all_pmcc_files, imageWidth, imageHeight, [fixed_layer], args.output_dir, args.jar_file, conf)

