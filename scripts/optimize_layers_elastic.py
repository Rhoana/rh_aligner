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

def optimize_layers_elastic(tile_files, corr_files, image_width, image_height, fixed_layers, out_dir, jar_file, conf=None):
    conf_args = utils.conf_args(conf, 'OptimizeLayersElastic')

    fixed_str = ""
    if fixed_layers != None:
        fixed_str = "--fixedLayers {0}".format(" ".join(map(str, fixed_layers)))

    java_cmd = 'java -Xmx4g -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.OptimizeLayersElastic --tilespecFiles {1} --corrFiles {2} \
            --fixedLayers {3} --imageWidth {4} --imageHeight {5} --targetDir {6} {7}'.format(
        jar_file,
        " ".join(utils.path2url(f) for f in tile_files),
        " ".join(utils.path2url(f) for f in corr_files),
        fixed_str,
        int(image_width),
        int(image_height),
        out_dir,
        conf_args)
    print "Executing: {0}".format(java_cmd)
    call(java_cmd, shell=True) # w/o shell=True it seems that the env-vars are not set

def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over the tilespecs in a file, computing matches for each overlapping tile.')
    parser.add_argument('tile_files', metavar='tile_files', type=str, nargs='+',
                        help='the list of tile spec files to align')
    parser.add_argument('corr_files', metavar='corr_files', type=str, nargs='+',
                        help='the list of corr spec files that contain the matched layers')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='an output directory that will include the aligned sections tiles (default: .)',
                        default='./')
    parser.add_argument('-w', '--image_width', type=int, 
                        help='the width of the image (used for creating the mesh)')
    parser.add_argument('-h', '--image_height', type=int, 
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


    args = parser.parse_args()

    optimize_layers_elastic(args.tile_files, args.corr_files, \
        args.image_width, args.image_height, args.fixed_layers, args.output_dir, args.jar_file, \
        conf=utils.conf_args_from_file(args.conf_file_name, "OptimizeLayersElastic"))

if __name__ == '__main__':
    main()

