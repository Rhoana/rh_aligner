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



def match_single_sift_features_and_filter(tiles_file, features_file1, features_file2, jar, out_fname, index_pair, conf_fname=None):
    tiles_url = utils.path2url(os.path.abspath(tiles_file))

    conf_args = utils.conf_args_from_file(conf_fname, 'MatchSiftFeaturesAndFilter')

    java_cmd = 'java -Xmx3g -XX:ParallelGCThreads=1 -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.MatchSiftFeaturesAndFilter \
            --tilespecfile {1} --featurefile1 {2} --featurefile2 {3} --indices {4} --targetPath {5} {6}'.format(
        jar,
        tiles_url,
        features_file1,
        features_file2,
        "{}:{}".format(index_pair[0], index_pair[1]),
        out_fname,
        conf_args)
    utils.execute_shell_command(java_cmd)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over the tilespecs in a file, computing matches for each overlapping tile.')
    parser.add_argument('tiles_file', metavar='tiles_file', type=str,
                        help='the json file of tilespecs')
    parser.add_argument('features_file1', metavar='features_file1', type=str,
                        help='a file that contains the features json file of the first tile')
    parser.add_argument('features_file2', metavar='features_file2', type=str,
                        help='a file that contains the features json file of the second tile')
    parser.add_argument('index_pair', metavar='index_pair', type=str,
                        help='a colon separated indices of the tiles in the tilespec file that correspond to the feature files that need to be matched')
    parser.add_argument('-o', '--output_file', type=str, 
                        help='an output file name where the correspondent_spec file will be (default: ./matched_sifts.json)',
                        default='./matched_sifts.json')
    parser.add_argument('-j', '--jar_file', type=str,
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')
    parser.add_argument('-c', '--conf_file_name', type=str, 
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)
    parser.add_argument('-w', '--wait_time', type=int, 
                        help='the time to wait since the last modification date of the features_file (default: None)',
                        default=0)


    args = parser.parse_args()


    index_pair = args.index_pair.split(':')

    utils.wait_after_file(args.features_file1, args.wait_time)
    utils.wait_after_file(args.features_file2, args.wait_time)

    match_single_sift_features_and_filter(args.tiles_file, args.features_file1, args.features_file2,
        args.jar_file, args.output_file, index_pair, conf_fname=args.conf_file_name)


if __name__ == '__main__':
    main()

