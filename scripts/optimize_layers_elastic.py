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

def optimize_layers_elastic(tile_files, corr_files, image_width, image_height, fixed_layers, out_dir, jar_file, conf=None, skip_layers=None, threads_num=4):
    conf_args = utils.conf_args_from_file(conf, 'OptimizeLayersElastic')

    fixed_str = ""
    if fixed_layers != None:
        fixed_str = "--fixedLayers {0}".format(" ".join(map(str, fixed_layers)))

    skip_str = ""
    if skip_layers != None:
        skip_str = "--skipLayers {0}".format(skip_layers)


    java_cmd = 'java -Xmx9g -XX:ParallelGCThreads=1 -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.OptimizeLayersElastic --tilespecFiles {1} --corrFiles {2} \
            {3} --imageWidth {4} --imageHeight {5} --threads {6} {7} --targetDir {8} {9}'.format(
        jar_file,
        " ".join(utils.path2url(f) for f in tile_files),
        " ".join(utils.path2url(f) for f in corr_files),
        fixed_str,
        int(image_width),
        int(image_height),
        threads_num,
        skip_str,
        out_dir,
        conf_args)
    utils.execute_shell_command(java_cmd)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over the tilespecs in a file, computing matches for each overlapping tile.')
    parser.add_argument('--tile_files', metavar='tile_files', type=str, nargs='+', required=True,
                        help='the list of tile spec files to align')
    parser.add_argument('--corr_files', metavar='corr_files', type=str, nargs='+', required=True,
                        help='the list of corr spec files that contain the matched layers')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='an output directory that will include the aligned sections tiles (default: .)',
                        default='./')
    parser.add_argument('-W', '--image_width', type=float, 
                        help='the width of the image (used for creating the mesh)')
    parser.add_argument('-H', '--image_height', type=float, 
                        help='the height of the image (used for creating the mesh)')
    parser.add_argument('-f', '--fixed_layers', type=str, nargs='+',
                        help='a space separated list of fixed layer IDs (default: None)',
                        default=None)
    parser.add_argument('-j', '--jar_file', type=str,
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')
    parser.add_argument('-c', '--conf_file_name', type=str, 
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)
    parser.add_argument('-t', '--threads_num', type=int, 
                        help='the number of threads to use (default: 1)',
                        default=1)
    parser.add_argument('-s', '--skip_layers', type=str, 
                        help='the range of layers (sections) that will not be processed e.g., "2,3,9-11,18" (default: no skipped sections)',
                        default=None)


    args = parser.parse_args()

    print "tile_files: {0}".format(args.tile_files)
    print "corr_files: {0}".format(args.corr_files)

    optimize_layers_elastic(args.tile_files, args.corr_files, \
        args.image_width, args.image_height, args.fixed_layers, args.output_dir, args.jar_file, \
        conf=args.conf_file_name, 
        skip_layers=args.skip_layers, threads_num=args.threads_num)

if __name__ == '__main__':
    main()

