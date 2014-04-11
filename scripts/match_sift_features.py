import sys
import os
import glob
import argparse
from subprocess import call
from bounding_box import BoundingBox
import json
import itertools

# common functions



def match_multiple_sift_features(tiles_file, features_file, index_pairs, jar, working_dir):
    match_out_file = '{0}_matchFeatures.json'.format(os.path.basename(tiles_file).replace('.json', ''))
    match_out_file = os.path.join(working_dir, match_out_file)
    java_cmd = 'java -cp "{0}" org.janelia.alignment.MatchSiftFeatures --featurefile {1} {2} --targetPath {3}'.format(
        jar,
        features_file,
        " ".join("--indices {}:{}".format(a, b) for a, b in index_pairs),
        match_out_file)
    print "Executing: {0}".format(java_cmd)
    call(java_cmd, shell=True) # w/o shell=True it seems that the env-vars are not set


def load_data_files(tile_file, features_file):
    with open(tile_file, 'r') as data_file:
        tilespecs = json.load(data_file)

    with open(features_file) as data_file:
        features = json.load(data_file)

    return tilespecs, {ft["mipmapLevels"]["0"]["imageUrl"] : idx for idx, ft in enumerate(features)}


def match_sift_features(tiles_file, features_file, working_dir, jar_file):
    # create a workspace directory if not found
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)


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
        bbox1 = BoundingBox(" ".join(str(b) for b in pair[0]["bbox"]))
        bbox2 = BoundingBox(" ".join(str(b) for b in pair[1]["bbox"]))
        if bbox1.overlap(bbox2):
            imageUrl1 = pair[0]["mipmapLevels"]["0"]["imageUrl"]
            imageUrl2 = pair[1]["mipmapLevels"]["0"]["imageUrl"]
            print "Matching sift of tiles: {0} and {1}".format(imageUrl1, imageUrl2)
            idx1 = feature_indices[imageUrl1]
            idx2 = feature_indices[imageUrl2]
            indices.append((idx1, idx2))
    match_multiple_sift_features(tiles_file, features_file, indices, jar_file, working_dir)

def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over the tilespecs in a file, computing matches for each overlapping tile.')
    parser.add_argument('tiles_file', metavar='tiles_file', type=str,
                        help='the json file of tilespecs')
    parser.add_argument('features_file', metavar='features_file', type=str,
                        help='the json file of features')
    parser.add_argument('-w', '--workspace_dir', type=str,
                        help='a directory where the output files will be kept (default: ./temp)',
                        default='./temp')
    parser.add_argument('-j', '--jar_file', type=str,
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')

    args = parser.parse_args()

    match_sift_features(args.tiles_file, args.features_file, args.workspace_dir, args.jar_file)

if __name__ == '__main__':
    main()

