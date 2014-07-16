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

def match_layers_by_max_pmcc(jar_file, tiles_file1, tiles_file2, models_file, image_width, image_height, fixed_layers, out_fname, conf=None, threads_num=1):
    conf_args = utils.conf_args(conf, 'MatchLayersByMaxPMCC')

    fixed_str = ""
    if fixed_layers != None:
        fixed_str = "--fixedLayers {0}".format(" ".join(map(str, fixed_layers)))

    java_cmd = 'java -Xmx6g -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.MatchLayersByMaxPMCC --inputfile1 {1} --inputfile2 {2} \
            --modelsfile1 {3} --imageWidth {4} --imageHeight {5} --threads {6} --targetPath {7} {8} {9}'.format(
        jar_file,
        utils.path2url(tiles_file1),
        utils.path2url(tiles_file2),
        utils.path2url(models_file),
        int(image_width),
        int(image_height),
        threads_num,
        out_fname,
        fixed_str,
        conf_args)
    print "Executing: {0}".format(java_cmd)
    call(java_cmd, shell=True) # w/o shell=True it seems that the env-vars are not set

def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over the tilespecs in a file, computing matches for each overlapping tile.')
    parser.add_argument('tiles_file1', metavar='tiles_file1', type=str,
                        help='the first layer json file of tilespecs')
    parser.add_argument('tiles_file2', metavar='tiles_file2', type=str,
                        help='the second layer json file of tilespecs')
    parser.add_argument('models_file', metavar='models_file', type=str,
                        help='a json file that contains the model to transform tiles from tiles_file1 to tiles_file2')
    parser.add_argument('-o', '--output_file', type=str, 
                        help='an output correspondent_spec file, that will include the sift features for each tile (default: ./pmcc_match.json)',
                        default='./pmcc_match.json')
    parser.add_argument('-W', '--image_width', type=float, 
                        help='the width of the image (used for creating the mesh)')
    parser.add_argument('-H', '--image_height', type=float, 
                        help='the height of the image (used for creating the mesh)')
    parser.add_argument('-f', '--fixed_layers', type=int, nargs='+',
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


    args = parser.parse_args()

    match_layers_by_max_pmcc(args.jar_file, args.tiles_file1, args.tiles_file2, \
        args.models_file, args.image_width, args.image_height, \
        args.fixed_layers, args.output_file, \
        conf=utils.conf_args_from_file(args.conf_file_name, "MatchLayersByMaxPMCC"), \
        threads_num=args.threads_num)

if __name__ == '__main__':
    main()

