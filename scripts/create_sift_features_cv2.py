# Takes a tilespec and an index of a tile, and creates the sift features of that tile
#
# requires:
# - cv2

import sys
import os
import argparse
import utils
import cv2
import json
import numpy as np
import h5py


def create_sift_features(tiles_fname, out_fname, index, conf_fname=None):

    # load tilespecs files
    tilespecs = utils.load_tilespecs(tiles_fname)
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

    print "Found {} features".format(len(descs))
    # Save the features

    print "Saving sift features at: {}".format(out_fname)
    with h5py.File(out_fname, 'w') as hf:
        hf.create_dataset("imageUrl",
                            data=np.array(image_path.encode("utf-8"), dtype='S'))
        hf.create_dataset("pts/responses", data=np.array([p.response for p in pts], dtype=np.float32))
        hf.create_dataset("pts/locations", data=np.array([p.pt for p in pts], dtype=np.float32))
        hf.create_dataset("pts/sizes", data=np.array([p.size for p in pts], dtype=np.float32))
        hf.create_dataset("pts/octaves", data=np.array([p.octave for p in pts], dtype=np.float32))
        hf.create_dataset("descs", data=descs)

    # features = [{"location" : (np.array(k.pt) * 2).tolist(),
    #              "response" : k.response,
    #              "scale" : k.size,
    #              "descriptor" : d.tolist()} for k, d in zip(pts, descs)]

    # out_data = [{
    #     "mipmapLevels" : {
    #         "0" : {
    #             "imageUrl" : image_path,
    #             "featureList" : features
    #         }
    #     },
    #     "mipmapLevel" : 0
    # }]
    # with open(out_fname, 'w') as out:
    #     json.dump(out_data, out, indent=4)




def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over a directory that contains json files, \
        and creates the sift features of each file. \
        The output is either in the same directory or in a different, user-provided, directory \
        (in either case, we use a different file name).')
    parser.add_argument('tiles_fname', metavar='tiles_json', type=str, 
                        help='a tile_spec file that contains the images to create sift features for, in json format')
    parser.add_argument('index', metavar='index', type=int, 
                        help='the index of the tile in the tilespec that needs to be computed')
    parser.add_argument('-o', '--output_file', type=str, 
                        help='an output feature_spec file, that will include the sift features for all tiles (default: ./siftFeatures.json)',
                        default='./siftFeatures.json')
    parser.add_argument('-c', '--conf_file_name', type=str, 
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)


    args = parser.parse_args()

    try:
        create_sift_features(args.tiles_fname, args.output_file, args.index, conf_fname=args.conf_file_name)
    except:
        sys.exit("Error while executing: {0}".format(sys.argv))

if __name__ == '__main__':
    main()

