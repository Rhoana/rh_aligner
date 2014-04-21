# Iterates over a directory that contains Tile-Spec json files (one for each tile), and matches every two tiles that overlap by max PMCC.
# The output is either in the same directory or in a different, user-provided, directory
# (in either case, we use a different file name)
#
# requires:
# - java (executed from the command line)
# - 

import sys
import os
import glob
import argparse
from subprocess import call
from bounding_box import BoundingBox
import json
import itertools
import urllib
import urlparse

# common functions

def path2url(path):
    return urlparse.urljoin(
        'file:', urllib.pathname2url(os.path.abspath(path)))


def match_two_tiles_by_max_pmcc_features(tiles_fname, index_pairs, jar, out_fname):
    java_cmd = 'java -Xmx6g -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.MatchByMaxPMCC --inputfile {1} {2} --targetPath {3}'.format(\
        jar, tiles_fname,
        " ".join("--indices {}:{}".format(a, b) for a, b in index_pairs),
        out_fname)
    print "Executing: {0}".format(java_cmd)
    call(java_cmd, shell=True) # w/o shell=True it seems that the env-vars are not set


def load_entire_data(tile_file):
    # Loads the entire collection of tile spec files, and returns it with an index for each tile

    # load tile_file (json)
    tile_file = tile_file.replace('file://', '')
    with open(tile_file, 'r') as data_file:
        tilespecs = json.load(data_file)

    for idx, tl in enumerate(tilespecs):
        tl['idx'] = idx

    print tilespecs
    return tilespecs



def match_by_max_pmcc(tiles_fname, out_fname, jar_file):

    tiles = load_entire_data(tiles_fname)

    # TODO: add all tiles to a kd-tree so it will be faster to find overlap between tiles

    # iterate over the tiles, and for each tile, find intersecting tiles that overlap,
    # and match their features
    # Nested loop:
    #    for each tile_i in range[0..N):
    #        for each tile_j in range[tile_i..N)]
    indices = []
    for pair in itertools.combinations(tiles, 2):
        # if the two tiles intersect, match them
        bbox1 = BoundingBox.fromList(pair[0]['bbox'])
        bbox2 = BoundingBox.fromList(pair[1]['bbox'])
        if bbox1.overlap(bbox2):
            print "Matching by max pmcc tiles: {0} and {1}".format(pair[0], pair[1])
            idx1 = pair[0]['idx']
            idx2 = pair[1]['idx']
            indices.append((idx1, idx2))
        #else:
        #    print "Tiles: {0} and {1} do not overlap, so no matching is done".format(pair[0], pair[1])

    match_two_tiles_by_max_pmcc_features(tiles_fname, indices, jar_file, out_fname)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over a all tiles in a Tile-Spec json file,\
        and matches every two tiles that overlap by max PMCC.')
    parser.add_argument('tiles_fname', metavar='tiles_fname', type=str, 
                        help='a tile_spec file that contains the images that need to be matched')
    parser.add_argument('-o', '--output_file', type=str, 
                    help='an output correspondent_spec file (default: ./matchesPMCC.json)',
                    default="./matchesPMCC.json")
    parser.add_argument('-j', '--jar_file', type=str, 
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')

    args = parser.parse_args()

    #print args

    match_by_max_pmcc(args.tiles_fname, args.output_file, args.jar_file)

if __name__ == '__main__':
    main()

