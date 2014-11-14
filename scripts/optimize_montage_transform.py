# Iterates over a directory that contains correspondence list json files, and optimizes the montage by perfroming the transform on each file.
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


def optimize_montage_transform(correspondence_file, tilespec_file, fixed_tiles, output_file, jar_file, conf_fname=None):

    corr_url = utils.path2url(correspondence_file)
    tiles_url = utils.path2url(tilespec_file)
    conf_args = utils.conf_args(conf_fname, 'OptimizeMontageTransform')

    fixed_str = ""
    if fixed_tiles != None:
        fixed_str = "--fixedTiles {0}".format(" ".join(map(str, fixed_tiles)))

    java_cmd = 'java -Xmx4g -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.OptimizeMontageTransform --inputfile {1} --tilespecfile {2} {3} --targetPath {4} {5}'.format(\
        jar_file, corr_url, tiles_url, fixed_str, output_file, conf_args)
    utils.execute_shell_command(java_cmd)




def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Takes a correspondence list json file, \
        and optimizes the montage by perfroming the transform on each tile in the file.')
    parser.add_argument('correspondence_file', metavar='correspondence_file', type=str, 
                        help='a correspondence_spec file')
    parser.add_argument('tilespec_file', metavar='tilespec_file', type=str, 
                        help='a tilespec file containing all the tiles')
    parser.add_argument('-o', '--output_file', type=str, 
                        help='the output file',
                        default='./opt_montage_transform.json')
    parser.add_argument('-f', '--fixed_tiles', type=str, nargs='+',
                        help='a space separated list of fixed tile indices (default: 0)',
                        default="0")
    parser.add_argument('-j', '--jar_file', type=str, 
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')
    parser.add_argument('-c', '--conf_file_name', type=str, 
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)


    args = parser.parse_args()

    #print args

    try:
        optimize_montage_transform(args.correspondence_file, args.tilespec_file, args.fixed_tiles, args.output_file, args.jar_file, \
            conf_fname=args.conf_file_name)
    except:
        print "Error while executing: {0}".format(sys.argv)
        print "Exiting"
        sys.exit(1)

if __name__ == '__main__':
    main()

