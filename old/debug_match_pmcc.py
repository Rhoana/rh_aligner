# Debugs the output of a matching pmcc between two layers.
# Produces the images of the given sections (if a match output is found)
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
import json


def read_layer_from_file(tiles_spec_fname):
    layer = None
    with open(tiles_spec_fname, 'r') as data_file:
        data = json.load(data_file)
    for tile in data:
        if tile['layer'] is None:
            print "Error reading layer in one of the tiles in: {0}".format(tiles_spec_fname)
            sys.exit(1)
        if layer is None:
            layer = tile['layer']
        if layer != tile['layer']:
            print "Error when reading tiles from {0} found inconsistent layers numbers: {1} and {2}".format(tiles_spec_fname, layer, tile['layer'])
            sys.exit(1)
    if layer is None:
        print "Error reading layers file: {0}. No layers found.".format(tiles_spec_fname)
        sys.exit(1)
    return int(layer)


def create_dir(path):
    # create a directory if not found
    if not os.path.exists(path):
        os.makedirs(path)


def call_java_debug_match_pmcc(input_file, out_dir, jar_file):
    input_url = utils.path2url(os.path.abspath(input_file))
    java_cmd = 'java -Xmx4g -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.DebugCorrespondence --inputfile {1} --targetDir {2}'.format(\
        jar_file, input_url, out_dir)
    utils.execute_shell_command(java_cmd)

def find_section_files(tiles_dir, section_num1, section_num2):
    # Iterate over the files in the directory and find the file name that includes the section
    fname1 = None
    fname2 = None
    for fname in glob.glob(os.path.join(tiles_dir, '*.json')):
        layer = read_layer_from_file(fname)
        if layer == section_num1:
            fname1 = fname
        if layer == section_num2:
            fname2 = fname
        if fname1 != None and fname2 != None:
            return [fname1, fname2]
    # Could not find both layers
    print "Error: could not find the tile spec files of sections {0} and {1}".format(section_num1, section_num2)
    return [None, None]

def get_matched_pmcc_file(work_dir, s1_fname, s2_fname):
    # Returns the matched pmcc file name
    matched_pmcc_dir = os.path.join(work_dir, "matched_pmcc")
    fname1_prefix = os.path.splitext(os.path.basename(s1_fname))[0]
    fname2_prefix = os.path.splitext(os.path.basename(s2_fname))[0]
    match_json = os.path.join(matched_pmcc_dir, "{0}_{1}_match_pmcc.json".format(fname1_prefix, fname2_prefix))
    return match_json


def debug_match_pmcc(tiles_dir, work_dir, out_dir, section_num1, section_num2, jar_file):
    fname1, fname2 = find_section_files(tiles_dir, section_num1, section_num2)
    if fname1 != None and fname2 != None:
        match_fname = get_matched_pmcc_file(work_dir, fname1, fname2)
        call_java_debug_match_pmcc(match_fname, out_dir, jar_file)



def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Debugs the output of a matching pmcc between two layers.\
        Produces the images of the given sections (if a match output is found)')
    parser.add_argument('--tiles_dir', metavar='tiles_dir', type=str, required=True,
                        help='a directory that contains all the tile_spec json files (each for a single section)')
    parser.add_argument('--work_dir', type=str, required=True, 
                        help='the working directory where all files from the driver are found')
    parser.add_argument('-s1', '--section_num1', type=int, required=True, 
                        help='the number of the first section')
    parser.add_argument('-s2', '--section_num2', type=int, required=True, 
                        help='the number of the second section')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='an output directory (default: ./matched_pmcc_debug)',
                        default='./matched_pmcc_debug')
    parser.add_argument('-j', '--jar_file', type=str, 
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')


    args = parser.parse_args()

    #print args

    create_dir(args.output_dir)
    debug_match_pmcc(args.tiles_dir, args.work_dir, args.output_dir, args.section_num1, args.section_num2, args.jar_file)

if __name__ == '__main__':
    main()

