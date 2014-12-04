import sys
import os
import glob
import argparse
from subprocess import call
import utils


def normalize_coordinates(tiles_fname, output_dir, jar_file):

    tiles_url = utils.path2url(tiles_fname)

    java_cmd = 'java -Xmx2g -XX:ParallelGCThreads=1 -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.NormalizeCoordinates --targetDir {1} {2}'.format(\
        jar_file, output_dir, tiles_url)
    utils.execute_shell_command(java_cmd)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Normalizes a single tilespecs file to a coordinate system starting from (0,0).')
    parser.add_argument('tiles_file', metavar='tiles_file', type=str,
                        help='a list of json files that need to be normalized or a directories of json files')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='an output directory (default: ./after_norm)',
                        default='./after_norm')
    parser.add_argument('-j', '--jar_file', type=str,
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')

    args = parser.parse_args()

    utils.create_dir(args.output_dir)

    normalize_coordinates(args.tiles_file, args.output_dir, args.jar_file)

if __name__ == '__main__':
    main()

