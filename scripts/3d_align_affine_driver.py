# A driver for running 3D alignment using the FijiBento alignment project
# The input is two tile spec files with their 2d alignment
# and each file has also a z axis (layer) index, and the output is a tile spec after 3D alignment
# Activates ComputeLayerSIFTFeaturs -> MatchLayersSIFTFeatures -> FilterRansac -> OptimizeLayersAffine
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
#from create_sift_features import create_sift_features
from create_meshes import create_meshes
from create_layer_sift_features import create_layer_sift_features
from match_layers_sift_features import match_layers_sift_features
from filter_ransac import filter_ransac
from optimize_layers_affine import optimize_layers_affine
from utils import path2url, write_list_to_file, create_dir, read_layer_from_file, parse_range, read_conf_args
from bounding_box import BoundingBox




# Command line parser
parser = argparse.ArgumentParser(description='A driver that does a 3D affine alignment of images.')
parser.add_argument('input_dir', metavar='input_dir', type=str, 
                    help='a directory that contains all the tile_spec files of all sections (each section already aligned and in a single tile_spec file) json format')
parser.add_argument('-w', '--workspace_dir', type=str, 
                    help='a directory where the output files of the different stages will be kept (default: ./work_dir)',
                    default='./work_dir')
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
parser.add_argument('--auto_add_model', action="store_true", 
                    help='automatically add the identity model, if a model is not found')
parser.add_argument('--from_layer', type=int, 
                    help='the layer to start from (inclusive, default: the first layer in the data)',
                    default=-1)
parser.add_argument('--to_layer', type=int, 
                    help='the last layer to render (inclusive, default: the last layer in the data)',
                    default=-1)
parser.add_argument('-s', '--skip_layers', type=str, 
                    help='the range of layers (sections) that will not be processed e.g., "2,3,9-11,18" (default: no skipped sections)',
                    default=None)
parser.add_argument('-M', '--manual_match', type=str, nargs="*",
                    help='pairs of layers (sections) that will need to be manually aligned (not part of the max_layer_distance) e.g., "2:10,7:21" (default: none)',
                    default=None)



args = parser.parse_args()

print args

# create a workspace directory if not found
create_dir(args.workspace_dir)

conf = None
if not args.conf_file_name is None:
    conf = args.conf_file_name


#after_bbox_dir = os.path.join(args.workspace_dir, "after_bbox")
#create_dir(after_bbox_dir)
sifts_dir = os.path.join(args.workspace_dir, "sifts")
create_dir(sifts_dir)
matched_sifts_dir = os.path.join(args.workspace_dir, "matched_sifts")
create_dir(matched_sifts_dir)
after_ransac_dir = os.path.join(args.workspace_dir, "after_ransac")
create_dir(after_ransac_dir)


all_layers = []
layer_to_sifts = {}
layer_to_ts_json = {}
layer_to_json_prefix = {}
layer_meshes_dir = {}

skipped_layers = parse_range(args.skip_layers)

bbox_suffix = "_bbox"

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

    # update the bbox of each section
    #after_bbox_json = os.path.join(after_bbox_dir, "{0}{1}.json".format(tiles_fname_prefix, bbox_suffix)) 
    #if not os.path.exists(after_bbox_json):
    #    print "Updating bounding box of {0}".format(tiles_fname_prefix)
    #    update_bbox(args.jar_file, tiles_fname, out_dir=after_bbox_dir, out_suffix=bbox_suffix)
    #bbox = read_bbox(after_bbox_json)


    # create the sift features of these tiles
    print "Computing sift features of {0}".format(tiles_fname_prefix)
    sifts_json = os.path.join(sifts_dir, "{0}_sifts.json".format(tiles_fname_prefix))
    if not os.path.exists(sifts_json):
        #create_layer_sift_features(after_bbox_json, sifts_json, args.jar_file, conf)
        create_layer_sift_features(tiles_fname, sifts_json, args.jar_file, conf=conf)
    layer_to_sifts[layer] = sifts_json
    layer_to_json_prefix[layer] = tiles_fname_prefix
    #layer_to_ts_json[layer] = after_bbox_json
    layer_to_ts_json[layer] = tiles_fname


# Verify that all the layers are there and that there are no holes
all_layers.sort()
for i in range(len(all_layers) - 1):
    if all_layers[i + 1] - all_layers[i] != 1:
        for l in range(all_layers[i] + 1, all_layers[i + 1]):
            if l not in skipped_layers:
                print "Error missing layer {} between: {} and {}".format(l, all_layers[i], all_layers[i + 1])
                sys.exit(1)

print "Found the following layers: {0}".format(all_layers)

print "All json files prefix are: {0}".format(layer_to_json_prefix)

# Set the middle layer as a fixed layer
fixed_layers = [ all_layers[len(all_layers)//2] ]

# Handle manual matches
# manual_matches = {}
# if args.manual_match is not None:
#     for match in args.manual_match:
#         # parse the manual match string
#         match_layers = [int(l) for l in match.split(':')]
#         # add a manual match between the lower layer and the higher layer
#         if min(match_layers) not in manual_matches.keys():
#             manual_matches[min(match_layers)] = []
#         manual_matches[min(match_layers)].append(max(match_layers))


# Match and optimize each two layers in the required distance
all_matched_sifts_files = []
all_model_files = []
for ei, i in enumerate(all_layers):
    # layers_to_process = min(i + args.max_layer_distance + 1, all_layers[-1] + 1) - i
    # to_range = range(1, layers_to_process)
    # # add manual matches
    # if i in manual_matches.keys():
    #     for second_layer in manual_matches[i]:
    #         diff_layers = second_layer - i
    #         if diff_layers not in to_range:
    #             to_range.append(diff_layers)
    # Process all matched layers
    # print "layers_to_process {0}".format(to_range[-1])
    matched_after_layers = 0
    j = 1
    while matched_after_layers < args.max_layer_distance:
        if ei + j >= len(all_layers):
            break

        if i in skipped_layers or (i+j) in skipped_layers:
            print "Skipping matching of layers {} and {}, because at least one of them should be skipped".format(i, i+j)
            j += 1
            continue

        fname1_prefix = layer_to_json_prefix[i]
        fname2_prefix = layer_to_json_prefix[i + j]

        # match the features of neighboring tiles
        match_json = os.path.join(matched_sifts_dir, "{0}_{1}_sift_matches.json".format(fname1_prefix, fname2_prefix))
        if not os.path.exists(match_json):
            print "Matching layers' sifts: {0} and {1}".format(i, i + j)
            match_layers_sift_features(layer_to_ts_json[i], layer_to_sifts[i], \
                layer_to_ts_json[i + j], layer_to_sifts[i + j], match_json, args.jar_file, conf)
        all_matched_sifts_files.append(match_json)

        # filter and ransac the matched points
        ransac_fname = os.path.join(after_ransac_dir, "{0}_{1}_filter_ransac.json".format(fname1_prefix, fname2_prefix))
        if not os.path.exists(ransac_fname):
            print "Filter-and-Ransac of layers: {0} and {1}".format(i, i + j)
            filter_ransac(match_json, path2url(layer_to_ts_json[i]), ransac_fname, args.jar_file, conf)
        all_model_files.append(ransac_fname)

        j += 1
        matched_after_layers += 1

# Optimize all layers to a single 3d image
all_ts_files = layer_to_ts_json.values()
create_dir(args.output_dir)

ts_list_file = os.path.join(args.workspace_dir, "all_ts_files.txt")
write_list_to_file(ts_list_file, all_ts_files)
matched_sifts_list_file = os.path.join(args.workspace_dir, "all_matched_sifts_files.txt")
write_list_to_file(matched_sifts_list_file, all_matched_sifts_files)
model_list_file = os.path.join(args.workspace_dir, "all_model_files.txt")
write_list_to_file(model_list_file, all_model_files)

optimize_layers_affine([ ts_list_file ], [ matched_sifts_list_file ], [ model_list_file ], 
    fixed_layers, args.output_dir, args.max_layer_distance,
    args.jar_file, conf, args.skip_layers, manual_matches=args.manual_match)

