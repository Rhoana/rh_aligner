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

def match_layers_sift_features(tiles_file1, features_file1, tiles_file2, features_file2, out_fname, jar_file, conf=None):
    # When matching layers, no need to take bounding box into account
    conf_args = utils.conf_args(conf, 'MatchLayersSiftFeatures')

    java_cmd = 'java -Xmx4g -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.MatchLayersSiftFeatures --tilespec1 {1} --featurefile1 {2} \
            --tilespec2 {3} --featurefile2 {4} --targetPath {5} {6}'.format(
        jar_file,
        utils.path2url(tiles_file1),
        utils.path2url(features_file1),
        utils.path2url(tiles_file2),
        utils.path2url(features_file2),
        out_fname,
        conf_args)
    print "Executing: {0}".format(java_cmd)
    call(java_cmd, shell=True) # w/o shell=True it seems that the env-vars are not set

def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over the tilespecs in a file, computing matches for each overlapping tile.')
    parser.add_argument('tiles_file1', metavar='tiles_file1', type=str,
                        help='the first layer json file of tilespecs')
    parser.add_argument('features_file1', metavar='features_file1', type=str,
                        help='the first layer json file of features')
    parser.add_argument('tiles_file2', metavar='tiles_file2', type=str,
                        help='the second layer json file of tilespecs')
    parser.add_argument('features_file2', metavar='features_file2', type=str,
                        help='the second layer json file of features')
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

    match_layers_sift_features(args.tiles_file1, args.features_file1, \
        args.tiles_file2, args.features_file2, args.output_file, args.jar_file, \
        conf=utils.conf_args_from_file(args.conf_file_name, "MatchLayersSiftFeatures"))

if __name__ == '__main__':
    main()

