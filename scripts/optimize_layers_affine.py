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

def optimize_layers_affine(tile_files, corr_files, model_files, fixed_layers, out_dir, max_layer_distance, jar_file, conf=None, skip_layers=None, threads_num=4, manual_matches=None):
    conf_args = utils.conf_args_from_file(conf, 'OptimizeLayersAffine')

    fixed_str = ""
    if fixed_layers != None:
        fixed_str = " ".join("--fixedLayers {0}".format(str(fixed_layer)) for fixed_layer in fixed_layers)

    skip_str = ""
    if skip_layers != None:
        skip_str = "--skipLayers {0}".format(skip_layers)

    manual_matches_str = ""
    if manual_matches is not None:
        manual_matches_str = " ".join("--manualMatches {}".format(a) for a in manual_matches)


    # Assuming that at least 4 threads will be allocated for this job, and increasing the number of gc threads to 4 will make it faster
    java_cmd = 'java -Xmx96g -XX:ParallelGCThreads={0} -Djava.awt.headless=true -cp "{1}" org.janelia.alignment.OptimizeLayersAffine --tilespecFiles {2} --corrFiles {3} \
            --modelFiles {4} {5} {6} --threads {7} --maxLayersDistance {8} {9} --targetDir {10} {11}'.format(
        utils.get_gc_threads_num(threads_num),
        jar_file,
        " ".join(utils.path2url(f) for f in tile_files),
        " ".join(utils.path2url(f) for f in corr_files),
        " ".join(utils.path2url(f) for f in model_files),
        fixed_str,
        manual_matches_str,
        threads_num,
        max_layer_distance,
        skip_str,
        out_dir,
        conf_args)
    utils.execute_shell_command(java_cmd)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Executes the affine optimization given sift matches and a corresponding model')
    parser.add_argument('--tile_files', metavar='tile_files', type=str, nargs='+', required=True,
                        help='the list of tile spec files to align')
    parser.add_argument('--corr_files', metavar='corr_files', type=str, nargs='+', required=True,
                        help='the list of corr spec files that contain the matched layers')
    parser.add_argument('--mode_files', metavar='corr_files', type=str, nargs='+', required=True,
                        help='the list of model spec files that contain the matched layers')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='an output directory that will include the aligned sections tiles (default: .)',
                        default='./')
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
    parser.add_argument('-d', '--max_layer_distance', type=int, 
                        help='the largest distance between two layers to be matched (default: 1)',
                        default=1)
    parser.add_argument('-M', '--manual_match', type=str, nargs="*",
                        help='pairs of layers (sections) that will need to be manually aligned (not part of the max_layer_distance) e.g., "2:10,7:21" (default: none)',
                        default=None)


    args = parser.parse_args()

    print "tile_files: {0}".format(args.tile_files)
    print "corr_files: {0}".format(args.corr_files)
    print "model_files: {0}".format(args.model_files)
    print "manual_match: {0}".format(args.manual_match)

    optimize_layers_affine(args.tile_files, args.corr_files, args.model_files,
        args.fixed_layers, args.output_dir, args.max_layer_distance,
        args.jar_file,
        conf=args.conf_file_name, 
        skip_layers=args.skip_layers, threads_num=args.threads_num,
        manual_matches=args.manual_match)

if __name__ == '__main__':
    main()

