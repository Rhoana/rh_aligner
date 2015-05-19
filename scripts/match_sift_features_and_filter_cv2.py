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


def ransac(matches, target_model_type, iterations, epsilon, min_inlier_ratio, min_num_inlier):
    model = Model.create_model(target_model_type)
    for i in xrange(iterations):
        # choose a minimal number of matches randomly
        min_matches = np.random.choice(matches, size=target_model_type["min_matches"], replace=False)
        # Try to fit them to the model
        if model.fit(min_matches):
            


def filter_matches(matches, target_model_type, iterations, epsilon, min_inlier_ratio, min_num_inlier):
    """Perform a RANSAC filtering given all the matches"""
    target_model, ransac_matches = ransac(matches, target_model_type, iterations, epsilon, min_inlier_ratio, min_num_inlier)
    filtered_inliers = filter_model(ransac_matches, target_model, inliers, max_trust, min_num_inliers)
    return filtered_inliers

def get_tilespec_transformation(tilespec):
    transforms = tilespec["transforms"]
    res = np.eye(3, dtype=np.float32)
    for t in transforms:
        if "TranslationModel2D" in t["className"]:
            dx, dy = [float(f) for f in t["dataString"].split()]
            res = res + np.array([[0, 0, dx], [0, 0, dy], [0, 0, 0]])
        else:
            print "Error: unknown transformation model: {}".format(t["className"])
            sys.exit(1)
    print res
    return res

def transform(trans, point):
    point = np.array(point)
    point = np.append(point, [1])
    trans_point = np.dot(trans, point)
    return trans_point.tolist()

def match_single_sift_features_and_filter(tiles_file, features_file1, features_file2, out_fname, index_pair, conf_fname=None):

    params = utils.conf_from_file(conf_fname, 'MatchSiftFeaturesAndFilter')
    rod = params["rod"]

    print "Matching sift features of tilespecs file: {}, indices: {}".format(tiles_file, index_pair)
    # load tilespecs files
    tilespecs = utils.load_tilespecs(tiles_file)
    ts1 = tilespecs[index_pair[0]]
    ts2 = tilespecs[index_pair[1]]

    # load feature files
    print "Loading sift features"
    _, pts1, _, _, descs1 = load_features_hdf5(features_file1)
    _, pts2, _, _, descs2 = load_features_hdf5(features_file2)

    # Match the features
    print "Matching sift features"
    matches = match_features(descs1, descs2, rod)

    # filter the matched features


    # Get the tilespec transformation
    print "Getting transformation"
    ts1_transform = get_tilespec_transformation(ts1)
    ts2_transform = get_tilespec_transformation(ts2)

    # save the output (matches)
    p1s = [pts1[[m[0].queryIdx for m in matches]]][0]
    p2s = [pts2[[m[0].trainIdx for m in matches]]][0]

    out_data = [{
        "mipmapLevel" : 0,
        "url1" : ts1["mipmapLevels"]["0"]["imageUrl"],
        "url2" : ts2["mipmapLevels"]["0"]["imageUrl"],
        "correspondencePointPairs" : [
            { "p1" : { "w": np.array(transform(ts1_transform, p1)[:2]).tolist(), "l": np.array([p1[0], p1[1]]).tolist() }, 
              "p2" : { "w": np.array(transform(ts2_transform, p2)[:2]).tolist(), "l": np.array([p2[0], p2[1]]).tolist() } } for p1, p2 in zip(p1s, p2s)
        ]
    }]

    print out_data[0]['correspondencePointPairs'][0]

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

