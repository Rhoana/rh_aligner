from ..common.bounding_box import BoundingBox
from ..common import utils
from ..common import ransac
import argparse
import json
import cv2
import h5py
import numpy as np
from rh_renderer.models import Transforms
import multiprocessing as mp
import logging
import re

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
logger_handler = logging.StreamHandler()
logger_handler.setLevel(logging.DEBUG)
logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)

# common functions


def load_features_hdf5(features_file):
    features_file = features_file.replace('file://', '')
    with h5py.File(features_file, 'r') as m:
        imageUrl = str(m["imageUrl"][...])
        locations = m["pts/locations"][...]
        responses = None  # m["pts/responses"][...]
        scales = None  # m["pts/scales"][...]
        descs = m["descs"][...]
    return imageUrl, locations, responses, scales, descs


def match_features(descs1, descs2, rod):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descs1, descs2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < rod*n.distance:
            good.append([m])

    return good


def get_tilespec_transformation(tilespec):
    transforms = tilespec["transforms"]
    # TODO - right now it only assumes a single transform
    t = Transforms.from_tilespec(transforms[0])
    return t


def save_empty_matches_file(out_fname, image_url1, image_url2):
    out_data = [{
        "mipmapLevel": 0,
        "url1": image_url1,
        "url2": image_url2,
        "correspondencePointPairs": [],
        "model": []
    }]

    logger.info("Saving matches into {}".format(out_fname))
    with open(out_fname, 'w') as out:
        json.dump(out_data, out, sort_keys=True, indent=4)


def dist_after_model(model, p1_l, p2_l):
    '''Compute the distance after applying the model to the
       given points (used for debugging)'''
    p1_l = np.array(p1_l)
    p2_l = np.array(p2_l)
    p1_l_new = model.apply(p1_l)
    delta = p1_l_new - p2_l
    return np.sqrt(np.sum(delta ** 2))

def match_single_pair(ts1, ts2, features_file1, features_file2, out_fname, rod, iterations, max_epsilon, min_inlier_ratio, min_num_inlier, model_index, max_trust, det_delta):
    # load feature files
    logger.info("Loading sift features")
    _, pts1, _, _, descs1 = load_features_hdf5(features_file1)
    _, pts2, _, _, descs2 = load_features_hdf5(features_file2)

    logger.info("Loaded {} features from file: {}".format(pts1.shape[0], features_file1))
    logger.info("Loaded {} features from file: {}".format(pts2.shape[0], features_file2))

    min_features_num = 5
    if pts1.shape[0] < min_features_num or pts2.shape[0] < min_features_num:
        logger.info("Less than {} features (even before overlap) of one of the tiles, saving an empty match file")
        save_empty_matches_file(out_fname, ts1["mipmapLevels"]["0"]["imageUrl"], ts2["mipmapLevels"]["0"]["imageUrl"])
        return

    # Get the tilespec transformation
    logger.info("Getting transformation")
    ts1_transform = get_tilespec_transformation(ts1)
    ts2_transform = get_tilespec_transformation(ts2)

    # filter the features, so that only features that are in the overlapping tile will be matches
    bbox1 = BoundingBox.fromList(ts1["bbox"])
    logger.info("bbox1 {}".format(bbox1.toStr()))
    bbox2 = BoundingBox.fromList(ts2["bbox"])
    logger.info("bbox2 {}".format(bbox2.toStr()))
    overlap_bbox = bbox1.intersect(bbox2).expand(offset=50)
    logger.info("overlap_bbox {}".format(overlap_bbox.toStr()))

    features_mask1 = overlap_bbox.contains(ts1_transform.apply(pts1))
    features_mask2 = overlap_bbox.contains(ts2_transform.apply(pts2))

    pts1 = pts1[features_mask1]
    pts2 = pts2[features_mask2]
    descs1 = descs1[features_mask1]
    descs2 = descs2[features_mask2]
    logger.info("Found {} features in the overlap from file: {}".format(pts1.shape[0], features_file1))
    logger.info("Found {} features in the overlap from file: {}".format(pts2.shape[0], features_file2))

    min_features_num = 5
    if pts1.shape[0] < min_features_num or pts2.shape[0] < min_features_num:
        logger.info("Less than {} features in the overlap of one of the tiles, saving an empty match file")
        save_empty_matches_file(out_fname, ts1["mipmapLevels"]["0"]["imageUrl"], ts2["mipmapLevels"]["0"]["imageUrl"])
        return

    # Match the features
    logger.info("Matching sift features")
    matches = match_features(descs1, descs2, rod)

    logger.info("Found {} possible matches between {} and {}".format(len(matches), features_file1, features_file2))

    # filter the matched features
    match_points = np.array([
        np.array([pts1[[m[0].queryIdx for m in matches]]][0]),
        np.array([pts2[[m[0].trainIdx for m in matches]]][0]) ])

    model, filtered_matches = ransac.filter_matches(match_points, model_index, iterations, max_epsilon, min_inlier_ratio, min_num_inlier, max_trust, det_delta)

    model_json = []
    if model is None:
        filtered_matches = [[], []]
    else:
        model_json = model.to_modelspec()

    # save the output (matches)
    out_data = [{
        "mipmapLevel": 0,
        "url1": ts1["mipmapLevels"]["0"]["imageUrl"],
        "url2": ts2["mipmapLevels"]["0"]["imageUrl"],
        "correspondencePointPairs": [
            { "p1": { "w": np.array(ts1_transform.apply(p1)[:2]).tolist(), "l": np.array([p1[0], p1[1]]).tolist() },
              "p2": { "w": np.array(ts2_transform.apply(p2)[:2]).tolist(), "l": np.array([p2[0], p2[1]]).tolist() },
              "dist_after_ransac": dist_after_model(model, p1, p2)
            } for p1, p2 in zip(filtered_matches[0], filtered_matches[1])
        ],
        "model": model_json
    }]

    logger.info("Saving matches into {}".format(out_fname))
    with open(out_fname, 'w') as out:
        json.dump(out_data, out, sort_keys=True, indent=4)

    return True


def match_single_sift_features_and_filter(tiles_file, features_file1, features_file2, out_fname, index_pair, conf_fname=None):

    params = utils.conf_from_file(conf_fname, 'MatchSiftFeaturesAndFilter')
    if params is None:
        params = {}
    rod = params.get("rod", 0.92)
    iterations = params.get("iterations", 1000)
    max_epsilon = params.get("maxEpsilon", 100.0)
    min_inlier_ratio = params.get("minInlierRatio", 0.01)
    min_num_inlier = params.get("minNumInliers", 7)
    model_index = params.get("modelIndex", 1)
    max_trust = params.get("maxTrust", 3)
    det_delta = params.get("detDelta", 0.3)

    logger.info("Matching sift features of tilespecs file: {}, mfovs-indices: {}".format(tiles_file, index_pair))
    # load tilespecs files
    indexed_tilespecs = utils.index_tilespec(utils.load_tilespecs(tiles_file))
    # Verify that the tiles are in the tilespecs (should be the case, unless they were filtered out)
    if index_pair[0] not in indexed_tilespecs:
        logger.info("The given mfov {} was not found in the tilespec: {}".format(index_pair[0], tiles_file))
        return
    if index_pair[1] not in indexed_tilespecs[index_pair[0]]:
        logger.info("The given tile_index {} in mfov {} was not found in the tilespec: {}".format(index_pair[1], index_pair[0], tiles_file))
        return
    if index_pair[2] not in indexed_tilespecs:
        logger.info("The given mfov {} was not found in the tilespec: {}".format(index_pair[2], tiles_file))
        return
    if index_pair[3] not in indexed_tilespecs[index_pair[2]]:
        logger.info("The given tile_index {} in mfov {} was not found in the tilespec: {}".format(index_pair[3], index_pair[2], tiles_file))
        return

    # The tiles should be part of the tilespecs, match them
    ts1 = indexed_tilespecs[index_pair[0]][index_pair[1]]
    ts2 = indexed_tilespecs[index_pair[2]][index_pair[3]]

    match_single_pair(ts1, ts2, features_file1, features_file2, out_fname, rod, iterations, max_epsilon, min_inlier_ratio, min_num_inlier, model_index, max_trust, det_delta)


def match_multiple_sift_features_and_filter(tiles_file, features_files_lst1, features_files_lst2, out_fnames, index_pairs, conf_fname=None, processes_num=1):

    params = utils.conf_from_file(conf_fname, 'MatchSiftFeaturesAndFilter')
    if params is None:
        params = {}
    rod = params.get("rod", 0.92)
    iterations = params.get("iterations", 1000)
    max_epsilon = params.get("maxEpsilon", 100.0)
    min_inlier_ratio = params.get("minInlierRatio", 0.01)
    min_num_inlier = params.get("minNumInliers", 7)
    model_index = params.get("modelIndex", 1)
    max_trust = params.get("maxTrust", 3)
    det_delta = params.get("detDelta", 0.3)

    assert(len(index_pairs) == len(features_files_lst1))
    assert(len(index_pairs) == len(features_files_lst2))
    assert(len(index_pairs) == len(out_fnames))

    logger.info("Creating a pool of {} processes".format(processes_num))
    pool = mp.Pool(processes=processes_num)

    indexed_tilespecs = utils.index_tilespec(utils.load_tilespecs(tiles_file))
    pool_results = []
    for i, index_pair in enumerate(index_pairs):
        features_file1 = features_files_lst1[i]
        features_file2 = features_files_lst2[i]
        out_fname = out_fnames[i]

        logger.info("Matching sift features of tilespecs file: {}, indices: {}".format(tiles_file, index_pair))
        # Verify that the tiles are in the tilespecs (should be the case, unless they were filtered out)
        if index_pair[0] not in indexed_tilespecs:
            logger.info("The given mfov {} was not found in the tilespec: {}".format(index_pair[0], tiles_file))
            continue
        if index_pair[1] not in indexed_tilespecs[index_pair[0]]:
            logger.info("The given tile_index {} in mfov {} was not found in the tilespec: {}".format(index_pair[1], index_pair[0], tiles_file))
            continue
        if index_pair[2] not in indexed_tilespecs:
            logger.info("The given mfov {} was not found in the tilespec: {}".format(index_pair[2], tiles_file))
            continue
        if index_pair[3] not in indexed_tilespecs[index_pair[2]]:
            logger.info("The given tile_index {} in mfov {} was not found in the tilespec: {}".format(index_pair[3], index_pair[2], tiles_file))
            continue

        # The tiles should be part of the tilespecs, match them
        ts1 = indexed_tilespecs[index_pair[0]][index_pair[1]]
        ts2 = indexed_tilespecs[index_pair[2]][index_pair[3]]

        res = pool.apply_async(match_single_pair, (ts1, ts2, features_file1, features_file2, out_fname, rod, iterations, max_epsilon, min_inlier_ratio, min_num_inlier, model_index, max_trust, det_delta))
        pool_results.append(res)

    # Verify that the returned values are okay (otherwise an exception will be shown)
    for res in pool_results:
        res.get()

    pool.close()
    pool.join()



def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over the tilespecs in a file, computing matches for each overlapping tile.')
    parser.add_argument('tiles_file', metavar='tiles_file', type=str,
                        help='the json file of tilespecs')
    parser.add_argument('features_file1', metavar='features_file1', type=str,
                        help='a file that contains the features json file of the first tile (if a single pair is matched) or a list of features json files (if multiple pairs are matched)')
    parser.add_argument('features_file2', metavar='features_file2', type=str,
                        help='a file that contains the features json file of the second tile (if a single pair is matched) or a list of features json files (if multiple pairs are matched)')
    parser.add_argument('--index_pairs', metavar='index_pairs', type=str, nargs='+',
                        help='a colon separated indices of the tiles in the tilespec file that correspond to the feature files that need to be matched. The format is [mfov1_index]_[tile_index]:[mfov2_index]_[tile_index]')
    parser.add_argument('-o', '--output_file', type=str,
                        help='an output file name where the correspondent_spec file will be (if a single pair is matched, default: ./matched_sifts.json) or a list of output files (if multiple pairs are matched, default: ./matched_siftsX.json)',
                        default='./matched_sifts.json')
    parser.add_argument('-c', '--conf_file_name', type=str,
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)
    parser.add_argument('-w', '--wait_time', type=int,
                        help='the time to wait since the last modification date of the features_file (default: None)',
                        default=0)
    parser.add_argument('-t', '--threads_num', type=int,
                        help='the number of threads (processes) to use (default: 1)',
                        default=1)

    args = parser.parse_args()
    print("args:", args)


    if len(args.index_pairs) == 1:
        utils.wait_after_file(args.features_file1, args.wait_time)
        utils.wait_after_file(args.features_file2, args.wait_time)
        
        m = re.match('([0-9]+)_([0-9]+):([0-9]+)_([0-9]+)', args.index_pairs[0])
        index_pair = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
        match_single_sift_features_and_filter(args.tiles_file, args.features_file1, args.features_file2,
                                              args.output_file, index_pair, conf_fname=args.conf_file_name)
    else: # More than one pair

        with open(args.features_file1, 'r') as f_fnames:
            features_files_lst1 = [fname.strip() for fname in f_fnames.readlines()]
        with open(args.features_file2, 'r') as f_fnames:
            features_files_lst2 = [fname.strip() for fname in f_fnames.readlines()]
        for feature_file in zip(features_files_lst1, features_files_lst2):
            utils.wait_after_file(feature_file[0], args.wait_time)
            utils.wait_after_file(feature_file[1], args.wait_time)
        with open(args.output_file, 'r') as f_fnames:
            output_files_lst = [fname.strip() for fname in f_fnames.readlines()]

        index_pairs = []
        for index_pair in args.index_pairs:
            m = re.match('([0-9]+)_([0-9]+):([0-9]+)_([0-9]+)', index_pair)
            index_pairs.append( (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))) )
        match_multiple_sift_features_and_filter(args.tiles_file, features_files_lst1, features_files_lst2,
                                                output_files_lst, index_pairs, conf_fname=args.conf_file_name,
                                                processes_num=args.threads_num)

    print("Done.")

if __name__ == '__main__':
    main()

