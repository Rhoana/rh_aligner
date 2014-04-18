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


def match_two_tiles_by_max_pmcc_features(tile1, tile2, jar, working_dir):
    fname1, ext1 = os.path.splitext(tile1['imageUrl'].split(os.path.sep)[-1])
    fname2, ext2 = os.path.splitext(tile2['imageUrl'].split(os.path.sep)[-1])
    match_out_file = '{0}_{1}_match_max_PMCC.json'.format(fname1, fname2)
    match_out_file = os.path.join(working_dir, match_out_file)
    #match_out_file = path2url(match_out_file)
    java_cmd = 'java -cp "{0}" org.janelia.alignment.MatchByMaxPMCC --inputfile1 {1} --inputfile2 {2} --targetPath {3}'.format(\
        jar, tile1['tileSpec'], tile2['tileSpec'], match_out_file)
    print "Executing: {0}".format(java_cmd)
    call(java_cmd, shell=True) # w/o shell=True it seems that the env-vars are not set


def load_entire_data(tile_files):
    # Loads the entire collection of tile spec files, and returns
    # a mapping of a tile->[imageUrl, tile-spec-file, bounding_box]
    tiles = {}
    for tile_file in tile_files:
        tile = {}

        # load tile_file (json)
        with open(tile_file, 'r') as data_file:
            data = json.load(data_file)

        tile['tileSpec'] = path2url(tile_file)
        tile['imageUrl'] = data[0]['mipmapLevels']['0']['imageUrl']
        tile['boundingBox'] = data[0]['boundingBox']
        tiles[tile['imageUrl']] = tile

    return tiles



def match_by_max_pmcc(tiles_dir, working_dir, jar_file):
    # create a workspace directory if not found
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)


    tile_files = glob.glob(os.path.join(tiles_dir, '*'))

    tiles = load_entire_data(tile_files)

    # TODO: add all tiles to a kd-tree so it will be faster to find overlap between tiles

    # iterate over the tiles, and for each tile, find intersecting tiles that overlap,
    # and match their features
    # Nested loop:
    #    for each tile_i in range[0..N):
    #        for each tile_j in range[tile_i..N)]
    for pair in itertools.combinations(tiles, 2):
        # if the two tiles intersect, match them
        bbox1 = BoundingBox(tiles[pair[0]]['boundingBox'])
        bbox2 = BoundingBox(tiles[pair[1]]['boundingBox'])
        if bbox1.overlap(bbox2):
            print "Matching by max pmcc tiles: {0} and {1}".format(pair[0], pair[1])
            match_two_tiles_by_max_pmcc_features(tiles[pair[0]], tiles[pair[1]], jar_file, working_dir)
        #else:
        #    print "Tiles: {0} and {1} do not overlap, so no matching is done".format(pair[0], pair[1])




def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over a directory that contains Tile-Spec json files (one for each tile),\
        and matches every two tiles that overlap by max PMCC.')
    parser.add_argument('tiles_dir', metavar='tiles_dir', type=str, 
                        help='a directory that contains tile_spec files')
    parser.add_argument('-w', '--workspace_dir', type=str, 
                        help='a directory where the output files will be kept (default: ./temp)',
                        default='./temp')
    parser.add_argument('-j', '--jar_file', type=str, 
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')

    args = parser.parse_args()

    #print args

    match_by_max_pmcc(args.tiles_dir, args.workspace_dir, args.jar_file)

if __name__ == '__main__':
    main()

