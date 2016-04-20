# Takes a tilespec and an index of a tile, and creates the sift features of that tile
#
# requires:
# - cv2

import sys
import os
import argparse
from ..common import utils
import cv2
import numpy as np
import h5py

def create_sift_features(tilespecs, out_fname, index, conf_fname=None):

    tilespec = tilespecs[index]

    # load the image
    image_path = tilespec["mipmapLevels"]["0"]["imageUrl"]
    image_path = image_path.replace("file://", "")
    if image_path.endswith(".jp2"):
        import glymur
        img_gray = glymur.Jp2k(image_path)[:] # load in full resolution
    else:
        img_gray = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    print "Computing sift features for image: {}".format(image_path)

    # compute features for the given index
    # detector = cv2.FeatureDetector_create("SIFT")
    # extractor = cv2.DescriptorExtractor_create("SIFT")
    # #print("Detecting keypoints...")
    # kp = detector.detect(img_gray)
    # #print("Computing descriptions...")
    # pts, descs = extractor.compute(img_gray, kp)
    sift = cv2.SIFT()
    pts, descs = sift.detectAndCompute(img_gray, None)
    if descs is None:
        descs = []
        pts = []

    descs = np.array(descs, dtype=np.uint8)

    # Save the features

    print "Saving {} sift features at: {}".format(len(descs), out_fname)
    with h5py.File(out_fname, 'w') as hf:
        hf.create_dataset("imageUrl",
                            data=np.array(image_path.encode("utf-8"), dtype='S'))
        hf.create_dataset("pts/responses", data=np.array([p.response for p in pts], dtype=np.float32))
        hf.create_dataset("pts/locations", data=np.array([p.pt for p in pts], dtype=np.float32))
        hf.create_dataset("pts/sizes", data=np.array([p.size for p in pts], dtype=np.float32))
        hf.create_dataset("pts/octaves", data=np.array([p.octave for p in pts], dtype=np.float32))
        hf.create_dataset("descs", data=descs)




def create_multiple_sift_features(tiles_fname, out_fnames, indices, conf_fname=None):

    # load tilespecs files
    tilespecs = utils.load_tilespecs(tiles_fname)

    for index, out_fname in zip(indices, out_fnames):
        create_sift_features(tilespecs, out_fname, index, conf_fname)




def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over a directory that contains json files, \
        and creates the sift features of each file. \
        The output is either in the same directory or in a different, user-provided, directory \
        (in either case, we use a different file name). \
        The order and number of the indices must match the output files.')
    parser.add_argument('tiles_fname', metavar='tiles_json', type=str, 
                        help='a tile_spec file that contains the images to create sift features for, in json format')
    parser.add_argument('-i', '--indices', type=int, nargs='+', 
                        help='the indices of the tiles in the tilespec that needs to be computed')
    parser.add_argument('-o', '--output_files', type=str, nargs='+', 
                        help='output feature_spec files list, each will include the sift features for the corresponding tile')
    parser.add_argument('-c', '--conf_file_name', type=str, 
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)


    args = parser.parse_args()
    print args

    assert(len(args.output_files) == len(args.indices))
    try:
        create_multiple_sift_features(args.tiles_fname, args.output_files, args.indices, conf_fname=args.conf_file_name)
    except:
        sys.exit("Error while executing: {0}".format(sys.argv))

if __name__ == '__main__':
    main()

