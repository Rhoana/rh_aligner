import sys
import os
import glob
import argparse
from subprocess import call
from bounding_box import BoundingBox
import json
import itertools
import utils

# common functions



def match_multiple_sift_features_and_filter(tiles_file, features_file, index_pairs, jar, out_fname, conf_fname=None):
    tiles_url = utils.path2url(os.path.abspath(tiles_file))

    conf_args = utils.conf_args_from_file(conf_fname, 'MatchSiftFeaturesAndFilter')

    java_cmd = 'java -Xmx4g -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.MatchSiftFeaturesAndFilter --tilespecfile {1} --featurefile {2} {3} --targetPath {4} {5}'.format(
        jar,
        tiles_url,
        features_file,
        " ".join("--indices {}:{}".format(a, b) for a, b in index_pairs),
        out_fname,
        conf_args)
    utils.execute_shell_command(java_cmd)


def load_data_files(tile_file, features_file):
    tile_file = tile_file.replace('file://', '')
    with open(tile_file, 'r') as data_file:
        tilespecs = json.load(data_file)

    features_file = features_file.replace('file://', '')
    with open(features_file) as data_file:
        features = json.load(data_file)

    return tilespecs, {ft["mipmapLevels"]["0"]["imageUrl"] : idx for idx, ft in enumerate(features)}


def match_sift_features_and_filter(tiles_file, features_file, out_fname, jar_file, conf_fname=None):

    tilespecs, feature_indices = load_data_files(tiles_file, features_file)
    for k, v in feature_indices.iteritems():
        print k, v

    # TODO: add all tiles to a kd-tree so it will be faster to find overlap between tiles
    # TODO: limit searches for matches to overlap area of bounding boxes

    # iterate over the tiles, and for each tile, find intersecting tiles that overlap,
    # and match their features
    # Nested loop:
    #    for each tile_i in range[0..N):
    #        for each tile_j in range[tile_i..N)]
    indices = []
    for pair in itertools.combinations(tilespecs, 2):
        # if the two tiles intersect, match them
        bbox1 = BoundingBox.fromList(pair[0]["bbox"])
        bbox2 = BoundingBox.fromList(pair[1]["bbox"])
        if bbox1.overlap(bbox2):
            imageUrl1 = pair[0]["mipmapLevels"]["0"]["imageUrl"]
            imageUrl2 = pair[1]["mipmapLevels"]["0"]["imageUrl"]
            print "Matching sift of tiles: {0} and {1}".format(imageUrl1, imageUrl2)
            idx1 = feature_indices[imageUrl1]
            idx2 = feature_indices[imageUrl2]
            indices.append((idx1, idx2))

    match_multiple_sift_features_and_filter(tiles_file, features_file, indices, jar_file, out_fname, conf_fname)

def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over the tilespecs in a file, computing matches for each overlapping tile.')
    parser.add_argument('tiles_file', metavar='tiles_file', type=str,
                        help='the json file of tilespecs')
    parser.add_argument('features_file', metavar='features_file', type=str,
                        help='the json file of features')
    parser.add_argument('-o', '--output_file', type=str, 
                        help='an output correspondent_spec file, that will include the sift features for each tile (default: ./matches.json)',
                        default='./matches.json')
    parser.add_argument('-j', '--jar_file', type=str,
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')
    parser.add_argument('-c', '--conf_file_name', type=str, 
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)


    args = parser.parse_args()


    try:
        match_sift_features_and_filter(args.tiles_file, args.features_file, args.output_file, args.jar_file, \
            conf_fname=args.conf_file_name)
    except:
        print "Error while executing: {0}".format(sys.argv)
        print "Exiting"
        sys.exit(1)


if __name__ == '__main__':
    main()

