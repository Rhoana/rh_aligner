import sys
import os
import glob
import argparse
from bounding_box import BoundingBox
import json
import itertools
import utils
import cv2
import h5py
import numpy as np
import copy
from models import Transforms
import ransac

# common functions


def load_features_hdf5(features_file):
    with h5py.File(features_file, 'r') as m:
        imageUrl = str(m["imageUrl"][...])
        locations = m["pts/locations"][...]
        responses = None#m["pts/responses"][...]
        scales = None#m["pts/scales"][...]
        descs = m["descs"][...]
    return imageUrl, locations, responses, scales, descs

def match_features(descs1, descs2, rod):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descs1, descs2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < rod*n.distance:
            good.append([m])

    return good





def get_tilespec_transformation(tilespec):
    transforms = tilespec["transforms"]
    # TODO - right now it only assumes a single transform
    t = Transforms.from_tilespec(transforms[0])
    return t


def match_single_sift_features_and_filter(tiles_file, features_file1, features_file2, out_fname, index_pair, conf_fname=None):

    params = utils.conf_from_file(conf_fname, 'MatchSiftFeaturesAndFilter')
    if params is None:
        params = {}
    rod = params.get("rod", 0.92)
    iterations = params.get("iterations", 1000)
    max_epsilon = params.get("maxEpsilon", 100.0)
    min_inlier_ratio = params.get("min_inlier_ratio", 0.01)
    min_num_inlier = params.get("min_num_inlier", 7)
    model_index = params.get("model_index", 1)
    max_trust = params.get("max_trust", 3)

    print "Matching sift features of tilespecs file: {}, indices: {}".format(tiles_file, index_pair)
    # load tilespecs files
    tilespecs = utils.load_tilespecs(tiles_file)
    ts1 = tilespecs[index_pair[0]]
    ts2 = tilespecs[index_pair[1]]

    # load feature files
    print "Loading sift features"
    _, pts1, _, _, descs1 = load_features_hdf5(features_file1)
    _, pts2, _, _, descs2 = load_features_hdf5(features_file2)

    print "Loaded {} features from file: {}".format(pts1.shape[0], features_file1)
    print "Loaded {} features from file: {}".format(pts2.shape[0], features_file2)

    # Get the tilespec transformation
    print "Getting transformation"
    ts1_transform = get_tilespec_transformation(ts1)
    ts2_transform = get_tilespec_transformation(ts2)

    # filter the features, so that only features that are in the overlapping tile will be matches
    bbox1 = BoundingBox.fromList(ts1["bbox"])
    print "bbox1", bbox1.toStr()
    bbox2 = BoundingBox.fromList(ts2["bbox"])
    print "bbox2", bbox2.toStr()
    overlap_bbox = bbox1.intersect(bbox2).expand(offset=50)
    print "overlap_bbox", overlap_bbox.toStr()
    features_mask1 = overlap_bbox.contains(ts1_transform.apply(pts1))
    features_mask2 = overlap_bbox.contains(ts2_transform.apply(pts2))

    pts1 = pts1[features_mask1]
    pts2 = pts2[features_mask2]
    descs1 = descs1[features_mask1]
    descs2 = descs2[features_mask2]
    print "Found {} features in the overlap from file: {}".format(pts1.shape[0], features_file1)
    print "Found {} features in the overlap from file: {}".format(pts2.shape[0], features_file2)


    # Match the features
    print "Matching sift features"
    matches = match_features(descs1, descs2, rod)

    print "Found {} possible matches between {} and {}".format(len(matches), features_file1, features_file2)

    # filter the matched features
    match_points = np.array([
        np.array([pts1[[m[0].queryIdx for m in matches]]][0]),
        np.array([pts2[[m[0].trainIdx for m in matches]]][0]) ])

    model, filtered_matches = ransac.filter_matches(match_points, model_index, iterations, max_epsilon, min_inlier_ratio, min_num_inlier, max_trust)

    model_json = []
    if model is not None:
        model_json = model.to_modelspec()

    # save the output (matches)
    p1s = [pts1[[m[0].queryIdx for m in matches]]][0]
    p2s = [pts2[[m[0].trainIdx for m in matches]]][0]

    out_data = [{
        "mipmapLevel" : 0,
        "url1" : ts1["mipmapLevels"]["0"]["imageUrl"],
        "url2" : ts2["mipmapLevels"]["0"]["imageUrl"],
        "correspondencePointPairs" : [
            { "p1" : { "w": np.array(ts1_transform.apply(p1)[:2]).tolist(), "l": np.array([p1[0], p1[1]]).tolist() }, 
              "p2" : { "w": np.array(ts2_transform.apply(p2)[:2]).tolist(), "l": np.array([p2[0], p2[1]]).tolist() } } for p1, p2 in zip(p1s, p2s)
        ],
        "model" : model_json
    }]


    print "Saving matches into {}".format(out_fname)
    with open(out_fname, 'w') as out:
        json.dump(out_data, out, sort_keys=True, indent=4)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over the tilespecs in a file, computing matches for each overlapping tile.')
    parser.add_argument('tiles_file', metavar='tiles_file', type=str,
                        help='the json file of tilespecs')
    parser.add_argument('features_file1', metavar='features_file1', type=str,
                        help='a file that contains the features json file of the first tile')
    parser.add_argument('features_file2', metavar='features_file2', type=str,
                        help='a file that contains the features json file of the second tile')
    parser.add_argument('index_pair', metavar='index_pair', type=str,
                        help='a colon separated indices of the tiles in the tilespec file that correspond to the feature files that need to be matched')
    parser.add_argument('-o', '--output_file', type=str, 
                        help='an output file name where the correspondent_spec file will be (default: ./matched_sifts.json)',
                        default='./matched_sifts.json')
    parser.add_argument('-c', '--conf_file_name', type=str, 
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)
    parser.add_argument('-w', '--wait_time', type=int, 
                        help='the time to wait since the last modification date of the features_file (default: None)',
                        default=0)


    args = parser.parse_args()


    index_pair = args.index_pair.split(':')

    utils.wait_after_file(args.features_file1, args.wait_time)
    utils.wait_after_file(args.features_file2, args.wait_time)

    match_single_sift_features_and_filter(args.tiles_file, args.features_file1, args.features_file2,
        args.output_file, index_pair, conf_fname=args.conf_file_name)


if __name__ == '__main__':
    main()

