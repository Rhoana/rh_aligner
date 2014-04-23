# Iterates over a directory that contains json files, and creates the sift features of each file.
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
import utils


def create_sift_features(tiles_fname, out_fname, jar_file, conf=None):

    tiles_url = utils.path2url(os.path.abspath(tiles_fname))
    conf_args = utils.conf_args(conf, 'ComputeSiftFeatures')
    # Compute the Sift features for each tile in the tile spec file
    java_cmd = 'java -Xmx4g -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.ComputeSiftFeatures --all --url {1} --targetPath {2} {3}'.format(\
        jar_file, tiles_url, out_fname, conf_args)
    print "Executing: {0}".format(java_cmd)
    call(java_cmd, shell=True) # w/o shell=True it seems that the env-vars are not set




def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over a directory that contains json files, \
        and creates the sift features of each file. \
        The output is either in the same directory or in a different, user-provided, directory \
        (in either case, we use a different file name).')
    parser.add_argument('tiles_fname', metavar='tiles_json', type=str, 
                        help='a tile_spec file that contains the images to create sift features for, in json format')
    parser.add_argument('-o', '--output_file', type=str, 
                        help='an output feature_spec file, that will include the sift features for all tiles (default: ./siftFeatures.json)',
                        default='./siftFeatures.json')
    parser.add_argument('-j', '--jar_file', type=str, 
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')

    args = parser.parse_args()

    #print args

    create_sift_features(args.tiles_fname, args.output_file, args.jar_file)

if __name__ == '__main__':
    main()

