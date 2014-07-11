# Update the bounding box of the given tile specs, by re-executing the transforms on all tiles

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

def load_tiles(tiles_spec_fname):
    all_bboxes = []
    with open(tiles_spec_fname, 'r') as data_file:
        data = json.load(data_file)
    for tile in data:
        tile_bbox = BoundingBox.fromList(tile['bbox'])
        all_bboxes.append(tile_bbox)
    return all_bboxes

def read_bbox(tiles_spec_fname):
    all_bboxes = load_tiles(tiles_spec_fname)
    # merge the bounding boxes to a single bbox
    if len(all_bboxes) > 0:
        ret_val = all_bboxes[0]
        for bbox in all_bboxes:
            ret_val.extend(bbox)
        return ret_val.toArray()
    return None

def update_bbox(jar_file, tilespec_file, out_dir=None, out_suffix=None, threads_num=None):
    out_str = ""
    if out_dir != None:
        out_str = "--targetDir {0}".format(out_dir)
    suffix_str = ""
    if out_suffix != None:
        suffix_str = "--suffix {0}".format(out_suffix)
    threads_str = ""
    if threads_num != None:
        threads_str = "--threads {0}".format(threads_num)

    java_cmd = 'java -Xmx6g -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.UpdateBoundingBox {1} {2} {3} {4}'.format(
        jar_file,
        out_str,
        suffix_str,
        threads_str,
        utils.path2url(tilespec_file))
    print "Executing: {0}".format(java_cmd)
    call(java_cmd, shell=True) # w/o shell=True it seems that the env-vars are not set




def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Receives a tile spec file name, and updates its bounding box.')
    parser.add_argument('tilespec_file', metavar='tilespec_file', type=str,
                        help='a tilespec json files (of a single section)')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='an output directory (default: ./)',
                        default='./')
    parser.add_argument('-s', '--suffix', type=str, 
                        help='a suffix to add to the output json file (default: _bbox)',
                        default='_bbox')
    parser.add_argument('-t', '--threads', type=int, 
                        help='the number of threads to use (default: the number of cores)',
                        default=None)
    parser.add_argument('-j', '--jar_file', type=str,
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')


    args = parser.parse_args()

    update_bbox(args.jar_file, args.tilespec_file, args.output_dir, args.suffix, args.threads)

if __name__ == '__main__':
    main()

