# Looks for "bad" rotations after the optimize_affine phase

import sys
import os
import glob
import argparse
import json
import math


def verify_tiles(filename, all_tiles):
    for i, tile in enumerate(all_tiles):
        transforms = tile['transforms']

        if len(transforms) == 0:
            print "{}: no transformations found for tile #{}".format(filename, i)
            return False

        # check each transform
        for transform in transforms:
            if transform["className"] == "mpicbg.trakem2.transform.RigidModel2D":
                rot = transform["dataString"].split(' ')[0]
                if abs(float(rot)) > 0.1:
                    print "{}: transformation with a large rotation was found for tile #{}".format(filename, i)
                    return False


    # Haven't found any reason to say that the transformation is bad, so it is probably good
    return True


def find_possible_bad_after_affine(affine_dir):

    # Get all input json files
    json_files = [jf for jf in (glob.glob(os.path.join(affine_dir, '*.json')))]

    good_files = 0

    for jf in json_files:
        with open(jf) as data_file:
            all_tiles = json.load(data_file)

        if verify_tiles(jf, all_tiles):
            good_files += 1

    print "Found {} good files out of {} files".format(good_files, len(json_files))

def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Given a directory of json files after the affine optimization, looks for possibly "bad" transformations')
    parser.add_argument('affine_dir', metavar='affine_dir', type=str, 
                        help='a directory that contains json files that are the output of the OptimizeMontageAffine class')


    args = parser.parse_args()

    #print args

    find_possible_bad_after_affine(args.affine_dir)

if __name__ == '__main__':
    main()

