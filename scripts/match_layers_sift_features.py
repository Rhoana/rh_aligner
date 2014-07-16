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

def match_layers_sift_features(tiles_file1, features_file1, tiles_file2, features_file2, out_fname, jar_file, conf=None, threads_num=4):
    # When matching layers, no need to take bounding box into account
    conf_args = utils.conf_args(conf, 'MatchSiftFeatures')

    java_cmd = 'java -Xmx4g -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.MatchLayersSiftFeatures --threads {1} --tilespec1 {2} \
            --featurefile1 {3} --tilespec2 {4} --featurefile2 {5} --targetPath {6} {7}'.format(
        jar_file,
        threads_num,
        utils.path2url(tiles_file1),
        utils.path2url(features_file1),
        utils.path2url(tiles_file2),
        utils.path2url(features_file2),
        out_fname,
        conf_args)
    utils.execute_shell_command(java_cmd)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over the tilespecs in a file, computing matches for each overlapping tile.')
    parser.add_argument('tiles_file1', metavar='tiles_file1', type=str,
                        help='the first layer json file containing tilespecs')
    parser.add_argument('features_file1', metavar='features_file1', type=str,
                        help='the first layer json file containing features')
    parser.add_argument('tiles_file2', metavar='tiles_file2', type=str,
                        help='the second layer json file containing tilespecs')
    parser.add_argument('features_file2', metavar='features_file2', type=str,
                        help='the second layer json file containing features')
    parser.add_argument('-o', '--output_file', type=str, 
                        help='an output correspondent_spec file, that will include the sift features for each tile (default: ./matches.json)',
                        default='./matches.json')
    parser.add_argument('-j', '--jar_file', type=str,
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')
    parser.add_argument('-c', '--conf_file_name', type=str, 
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)
    parser.add_argument('-t', '--threads_num', type=int, 
                        help='the number of threads to use (default: 1)',
                        default=1)


    args = parser.parse_args()

    match_layers_sift_features(args.tiles_file1, args.features_file1, \
        args.tiles_file2, args.features_file2, args.output_file, args.jar_file, \
        conf=utils.conf_args_from_file(args.conf_file_name, "MatchSiftFeatures"), threads_num=args.threads_num)

if __name__ == '__main__':
    main()

