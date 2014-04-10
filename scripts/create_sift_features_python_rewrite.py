# Iterates over a directory that contains json files, and creates the sift features of each file.
# The output is either in the same directory or in a different, user-provided, directory
# (in either case, we use a different file name)
#
# requires:
# - java (executed from the command line)
# - 
from __future__ import print_function
import sys
import os
import glob
import argparse
from subprocess import call
import urllib
try:   
    from urlparse import urljoin
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urljoin
    from urllib.parse import urlparse
import cv2
import numpy as np
import itertools
import json
import time


# common functions

def tilegen(image,tile_size=1024,overlap=0):
    for i in range(0,image.shape[0],tile_size-overlap):
        for j in range(0,image.shape[1],tile_size-overlap):
            yield i,j,image[i:i+tile_size,j:j+tile_size].copy()

def url2path(url):
    p = urlparse(url)
    return os.path.join(p.netloc, p.path)

def extract_sift(image_path):
    img = cv2.imread(image_path,cv2.cv.CV_LOAD_IMAGE_GRAYSCALE)
   
    feature_list = []
    #print("Calling tilegen...")
    for offset_i,offset_j,tile in tilegen(img):
        detector = cv2.FeatureDetector_create("SIFT")
        descriptor = cv2.DescriptorExtractor_create("SIFT")
        #print("Detecting keypoints...")
        kp = detector.detect(tile)
        #print("Computing descriptions...")
        kp, descs = descriptor.compute(tile,kp)

        #print("Updating feature list...")
        for k,d in zip(kp,descs):
            feature_list += [{"scale" : k.size,
                              "orientation" : k.angle,
                              "location" : [k.pt[0]+offset_j,k.pt[1]+offset_i],
                              "descriptor": d.tolist()}] 
            # cv2 uses [0]=x(column),[1]=y(rows)
        return feature_list

def compute_all_tiles_sift_features(tile_files, jar, working_dir):
    tInit = time.time()
    for tile_file in tile_files:
        fname, ext = os.path.splitext(tile_file.split(os.path.sep)[-1])
        sift_out_file = '{0}_siftFeatures.json'.format(fname)
        sift_out_file = os.path.join(working_dir, sift_out_file)

        print( "Executing: {0}".format(fname))
        
        output_list = []
        with open(tile_file) as f:
            tilespecs = json.load(f)
            for tilespec in tilespecs:
                print(tilespec)
                filename = tilespec["imageUrl"]
                image_path = url2path(filename)
                image_path = image_path[1:]
                image_path = image_path.replace("/","\\")
                feature_list = (extract_sift(image_path))
                tile_info = {"imageUrl" : filename,
                             "featureList": feature_list}
                output_list.append(tile_info)

            if len(output_list) > 0:
                with open(sift_out_file, 'w') as outfile:
                    json.dump(output_list, outfile, ensure_ascii=False)
                        
        tFinal = time.time()
        timeSpent = tFinal-tInit
        print("Time spent on this section = {0}".format(timeSpent))
        #sys.exit("Testing on one section done")


def create_sift_features(tiles_dir, working_dir, jar_file):
    # create a workspace directory if not found
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)


    tile_files = glob.glob(os.path.join(tiles_dir, '*.json'))

    # Compute the Sift features for each tile
    compute_all_tiles_sift_features(tile_files, jar_file, working_dir)

def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over a directory that contains json files, \
        and creates the sift features of each file. \
        The output is either in the same directory or in a different, user-provided, directory \
        (in either case, we use a different file name).')
    parser.add_argument('tiles_dir', metavar='tiles_dir', type=str, 
                        help='a directory that contains tile_spec files. Sift features will be extracted from each tile')
    parser.add_argument('-w', '--workspace_dir', type=str, 
                        help='a directory where the output files will be kept (default: ./temp)',
                        default='./temp')
    parser.add_argument('-j', '--jar_file', type=str, 
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')

    args = parser.parse_args()

    #print args

    create_sift_features(args.tiles_dir, args.workspace_dir, args.jar_file)

if __name__ == '__main__':
    main()
