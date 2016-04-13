from __future__ import print_function
import sys
import cv2
from renderer.tilespec_affine_renderer import TilespecAffineRenderer
import argparse
import utils
from bounding_box import BoundingBox
import models
import numpy as np
import json
from scipy.spatial import distance
from scipy import spatial
import time
import subprocess
import os

TEMPLATE_SIZE = 400

def view_post_pmcc_mfov(pmcc_matches_fname, matches_num, seed, scale, output_dir):

    # Load the preliminary matches
    with open(pmcc_matches_fname, 'r') as data_matches:
        pmcc_matches_data = json.load(data_matches)
    if len(pmcc_matches_data["pointmatches"]) == 0:
        print("No matches were found in pmcc-matching, aborting")
        return

    tiles_fname1 = pmcc_matches_data["tilespec1"]
    tiles_fname2 = pmcc_matches_data["tilespec2"]

    # Read the tilespecs
    ts1 = utils.load_tilespecs(tiles_fname1)
    ts2 = utils.load_tilespecs(tiles_fname2)
    indexed_ts1 = utils.index_tilespec(ts1)
    indexed_ts2 = utils.index_tilespec(ts2)


    # Create the (lazy) renderers for the two sections
    img1_renderer = TilespecAffineRenderer(ts1)
    img2_renderer = TilespecAffineRenderer(ts2)

    scale_transformation = np.array([
                                [ scale, 0., 0. ],
                                [ 0., scale, 0. ]
                            ])
    img1_renderer.add_transformation(scale_transformation)
    img2_renderer.add_transformation(scale_transformation)


    # Find a random number of points
    np.random.seed(seed)
    matches_idxs = np.random.choice(len(pmcc_matches_data["pointmatches"]), matches_num, replace=False)
    template_size = int(TEMPLATE_SIZE * scale)
    print("Actual template size: {}".format(template_size))

    utils.create_dir(output_dir)
    # save the matches thumbnails to the output dir
    for idx in matches_idxs:
        # rescale the matches
        p1 = np.array(pmcc_matches_data["pointmatches"][idx]["point1"])
        p2 = np.array(pmcc_matches_data["pointmatches"][idx]["point2"])
        print("Saving match {}: {} and {}".format(idx, p1, p2))
        p1_scaled = p1 * scale
        p2_scaled = p2 * scale
        out_fname_prefix = os.path.join(output_dir, 'pmcc_match_{}-{}_{}-{}'.format(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
        
        # Crop and save
        cropped_img1, _ = img1_renderer.crop(p1_scaled[0] - template_size, p1_scaled[1] - template_size, p1_scaled[0] + template_size, p1_scaled[1] + template_size)
        cv2.imwrite('{}_image1.jpg'.format(out_fname_prefix), cropped_img1)
        cropped_img2, _ = img2_renderer.crop(p2_scaled[0] - template_size, p2_scaled[1] - template_size, p2_scaled[0] + template_size, p2_scaled[1] + template_size)
        cv2.imwrite('{}_image2.jpg'.format(out_fname_prefix), cropped_img2)

        

def main():
    # print(sys.argv)
    # Command line parser
    parser = argparse.ArgumentParser(description='Given a between slices post pmcc-match json file (of a given mfov), renders a given number of templates around the matches and shows them as a montage on the screen. The matches are chosen randomly using a seed.')
    parser.add_argument('pmcc_matches_file', metavar='pmcc_matches_file', type=str,
                        help='a json file that contains the pmcc matches')
    parser.add_argument('-n', '--matches_num', type=int,
                        help='number of matches to show (default: 10)',
                        default=10)
    parser.add_argument('--seed', type=int,
                        help='a RNG seed (default: 7)',
                        default=7)
    parser.add_argument('-s', '--scale', type=float,
                        help='output scale (default: 0.1)',
                        default=0.1)
    parser.add_argument('-o', '--output_dir', type=str,
                        help='the directory where the output will be stored (default: ./view_pmcc_matches_output)',
                        default='./view_pmcc_matches_output')

    args = parser.parse_args()
    view_post_pmcc_mfov(args.pmcc_matches_file, args.matches_num, args.seed, args.scale, args.output_dir)

if __name__ == '__main__':
    main()
