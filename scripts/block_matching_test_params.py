# Executes the block matching parameters testing application.
# This script opens a user GUI, and therefore requires a screen to show the results to
# Uses a subset of the MatchLayersByMaxPMCC configuration parameters 
#
# TODO: Add threads support

import sys
import os
import argparse
import utils


# common functions

def block_matching_test_params(tile_files, jar_file, conf=None, threads_num=4):
    #conf_args = utils.conf_args(conf, 'MatchLayersByMaxPMCC')

    java_cmd = 'java -Xmx16g -XX:ParallelGCThreads=1 -cp "{0}" org.janelia.alignment.TestBlockMatchingParameters --tilespecFiles {1} \
            --threads {2} {3}'.format(
        jar_file,
        " ".join(utils.path2url(f) for f in tile_files),
        threads_num,
        conf)
    utils.execute_shell_command(java_cmd)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Executes the block matching parameters testing application.')
    parser.add_argument('--tile_files', metavar='tile_files', type=str, nargs='+', required=True,
                        help='the list of tile spec files to test')
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

    print "tile_files: {0}".format(args.tile_files)

    block_matching_test_params(args.tile_files, args.jar_file, \
        conf=utils.conf_args_from_file(args.conf_file_name, "MatchLayersByMaxPMCC"), threads_num=args.threads_num)

if __name__ == '__main__':
    main()

