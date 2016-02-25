import sys
import os.path
import os
import argparse
import utils
import random

from collections import defaultdict
import json

import glob
import progressbar
import numpy as np
import scipy.sparse as spp
from scipy.sparse.linalg import lsqr


def dist(p1, p2):
    return np.sqrt(((p1 - p2) ** 2).sum(axis=0))


def find_rotation(p1, p2, stepsize):
    U, S, VT = np.linalg.svd(np.dot(p1, p2.T))
    R = np.dot(VT.T, U.T)
    angle = stepsize * np.arctan2(R[1, 0], R[0, 0])
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle),  np.cos(angle)]])


def create_new_tilespec(old_ts_fname, rotations, translations, centers, out_fname):
    print("Optimization done, saving tilespec at: {}".format(out_fname))
    with open(old_ts_fname, 'r') as f:
        tilespecs = json.load(f)

    # Iterate over the tiles in the original tilespec
    for ts in tilespecs:
        img_url = ts["mipmapLevels"]["0"]["imageUrl"]
        # print "Transforming {}".format(img_url)
        if img_url not in rotations.keys():
            print "Flagging out tile {}, as no rotation was found".format(img_url)
            continue
        # Get 4 points of the old bounding box [top_left, top_right, bottom_left, bottom_right]
        old_bbox = [float(d) for d in ts["bbox"]]
        old_bbox_points = [
            np.array([ np.array([old_bbox[0]]), np.array([old_bbox[2]]) ]),
            np.array([ np.array([old_bbox[1]]), np.array([old_bbox[2]]) ]),
            np.array([ np.array([old_bbox[0]]), np.array([old_bbox[3]]) ]),
            np.array([ np.array([old_bbox[1]]), np.array([old_bbox[3]]) ]) ]
        # print "old_bbox:", old_bbox_points
        # convert the transformation according to the rotations data
        # compute new bbox with rotations (Rot * (pt - center) + center + trans)
        trans = np.array(translations[img_url])  # an array of 2 elements
        rot_matrix = np.matrix(rotations[img_url]).T  # a 2x2 matrix
        center = np.array(centers[img_url])  # an array of 2 elements
        transformed_points = [np.dot(rot_matrix, old_point - center) + center + trans for old_point in old_bbox_points]
        # print "transformed_bbox:", transformed_points
        min_xy = np.min(transformed_points, axis=0).flatten()
        max_xy = np.max(transformed_points, axis=0).flatten()
        new_bbox = [min_xy[0], max_xy[0], min_xy[1], max_xy[1]]
        # print "new_bbox", new_bbox
        # compute the global transformation of the tile
        # the translation part is just taking (0, 0) and moving it to the first transformed_point
        delta = np.asarray(transformed_points[0].T)[0]

        x, y = np.asarray((old_bbox_points[1] - old_bbox_points[0]).T)[0]
        new_x, new_y = np.asarray(transformed_points[1].T)[0]
        k = (y * (new_x - delta[0]) - x * (new_y - delta[1])) / (x**2 + y**2)
        h1 = (new_x - delta[0] - k*y)/x
        new_transformation = "{} {} {}".format(np.arccos(h1), delta[0], delta[1])
        # print "new_transformation:", new_transformation

        # Verify the result - for debugging (needs to be the same as the new bounding box)
        # new_matrix = np.array([ [h1, k, delta[0]],
        #                         [-k, h1, delta[1]],
        #                         [0.0, 0.0, 1.0]])
        # tile_points = [np.asarray((old_bbox_points[i] - old_bbox_points[0]).T)[0] for i in range(4)]
        # tile_points = [np.append(tile_point, [1.0], 0) for tile_point in tile_points]
        # after_trans = [np.dot(new_matrix, tile_point) for tile_point in tile_points]
        # print "tile 4 coordinates after_trans", after_trans

        # Set the transformation in the tilespec
        ts["transforms"] = [{
                "className": "mpicbg.trakem2.transform.RigidModel2D",
                "dataString": new_transformation
            }]

        ts["bbox"] = new_bbox

    # Save the new tilespecs
    with open(out_fname, 'w') as outjson:
        json.dump(tilespecs, outjson, sort_keys=True, indent=4)
        print('Wrote tilespec to {0}'.format(out_fname))


def optimize_2d_mfovs(tiles_fname, match_list_file, out_fname, conf_fname=None):
    # all matched pairs between point sets
    all_matches = {}
    # all points from a given tile
    all_pts = defaultdict(list)

    # load the list of files
    with open(match_list_file, 'r') as list_file:
        match_files = list_file.readlines()
    match_files = [fname.replace('\n', '').replace('file://', '') for fname in match_files]
    # print match_files

    # Load config parameters
    params = utils.conf_from_file(conf_fname, 'Optimize2Dmfovs')
    if params is None:
        params = {}
    maxiter = params.get("maxIterations", 1000)
    epsilon = params.get("maxEpsilon", 5)
    stepsize = params.get("stepSize", 0.1)
    damping = params.get("damping", 0.01)  # in units of matches per pair
    noemptymatches = params.get("noEmptyMatches", True)
    tilespec = json.load(open(tiles_fname, 'r'))

    # load the matches
    pbar = progressbar.ProgressBar()
    for f in pbar(match_files):
        data = json.load(open(f))
        # point arrays are 2xN
        pts1 = np.array([c["p1"]["w"] for c in data[0]["correspondencePointPairs"]]).T
        pts2 = np.array([c["p2"]["w"] for c in data[0]["correspondencePointPairs"]]).T
        url1 = data[0]["url1"]
        url2 = data[0]["url2"]
        if pts1.size > 0:
            all_matches[url1, url2] = (pts1, pts2)
            all_pts[url1].append(pts1)
            all_pts[url2].append(pts2)
        # If we want to add fake points when no matches are found
        elif noemptymatches:
            # Find the images in the tilespec
            mfov1, mfov2 = -1, -1
            for t in tilespec:
                if t['mipmapLevels']['0']['imageUrl'] == url1:
                    tile1 = t
                    mfov1 = t["mfov"]
                if t['mipmapLevels']['0']['imageUrl'] == url2:
                    tile2 = t
                    mfov2 = t["mfov"]
            if mfov1 == -1 or mfov2 == -1 or mfov1 != mfov2:
                continue

            # Determine the region of overlap between the two images
            overlapx_min = max(tile1['bbox'][0], tile2['bbox'][0])
            overlapx_max = max(tile1['bbox'][1], tile2['bbox'][1])
            overlapy_min = max(tile1['bbox'][2], tile2['bbox'][2])
            overlapy_max = max(tile1['bbox'][3], tile2['bbox'][3])
            obbox = [overlapx_min, overlapx_max, overlapy_min, overlapy_max]
            xrang, yrang = obbox[1] - obbox[0], obbox[3] - obbox[2]
            if xrang < 0 or yrang < 0:
                # The two areas do not overlap
                continue

            # Choose four random points in the overlap region - one from each quadrant
            xvals, yvals = [], []
            xvals.append(random.random() * xrang / 2 + obbox[0])
            xvals.append(random.random() * xrang / 2 + obbox[0] + xrang / 2)
            xvals.append(random.random() * xrang / 2 + obbox[0])
            xvals.append(random.random() * xrang / 2 + obbox[0] + xrang / 2)

            yvals.append(random.random() * yrang / 2 + obbox[2])
            yvals.append(random.random() * yrang / 2 + obbox[2])
            yvals.append(random.random() * yrang / 2 + obbox[2] + yrang / 2)
            yvals.append(random.random() * yrang / 2 + obbox[2] + yrang / 2)

            # Add these 4 points to a list of point pairs
            corpairs = []
            for i in range(0, len(xvals)):
                newpair = {}
                newpair['dist_after_ransac'] = 1.0
                newp1 = {'l': [xvals[i] - tile1['bbox'][0], yvals[i] - tile1['bbox'][2]], 'w': [xvals[i], yvals[i]]}
                newp2 = {'l': [xvals[i] - tile2['bbox'][0], yvals[i] - tile2['bbox'][2]], 'w': [xvals[i], yvals[i]]}
                newpair['p1'] = newp1
                newpair['p2'] = newp2
                corpairs.append(newpair)

            pts1 = np.array([c["p1"]["w"] for c in corpairs]).T
            pts2 = np.array([c["p2"]["w"] for c in corpairs]).T
            all_matches[url1, url2] = (pts1, pts2)
            all_pts[url1].append(pts1)
            all_pts[url2].append(pts2)

    # Find centers of each group of points
    centers = {k: np.mean(np.hstack(pts), axis=1, keepdims=True) for k, pts in all_pts.iteritems()}
    # a unique index for each url
    url_idx = {url: idx for idx, url in enumerate(all_pts)}

    prev_meanmed = np.inf

    T = defaultdict(lambda: np.zeros((2, 1)))
    R = defaultdict(lambda: np.eye(2))
    for iter in range(maxiter):
        # transform points by the current trans/rot
        trans_matches = {(k1, k2): (np.dot(R[k1], p1 - centers[k1]) + T[k1] + centers[k1],
                                    np.dot(R[k2], p2 - centers[k2]) + T[k2] + centers[k2])
                         for (k1, k2), (p1, p2) in all_matches.iteritems()}

        # mask off all points more than epsilon past the median
        diffs = {k: p2 - p1 for k, (p1, p2) in trans_matches.iteritems()}
        distances = {k: np.sqrt((d ** 2).sum(axis=0)) for k, d in diffs.iteritems()}
        masks = {k: d < (np.median(d) + epsilon) for k, d in distances.iteritems()}
        masked_matches = {k: (p1[:, masks[k]], p2[:, masks[k]]) for k, (p1, p2) in trans_matches.iteritems()}

        median_dists = [np.median(d) for d in distances.values()]
        medmed = np.median(median_dists)
        meanmed = np.mean(median_dists)
        maxmed = np.max(median_dists)
        print("med-med distance: {}, mean-med distance: {}  max-med: {}  SZ: {}".format(medmed, meanmed, maxmed, stepsize))
        if meanmed < prev_meanmed:
            stepsize *= 1.1
            if stepsize > 1:
                stepsize = 1
        else:
            stepsize *= 0.5
        prev_meanmed = meanmed

        # Find optimal translations
        #
        # Build a sparse matrix M of c/0/-c for differences between match sets,
        # where c is the size of each match set, and a vector D of sums of
        # differences, and then solve for T:
        #    M * T = D
        # to get the translations (independently in x and y).
        #
        # M is IxJ, I = number of match pairs, J = number of tiles
        # T is Jx2, D is Ix2  (2 for x, y)

        # two nonzero entries per match set
        rows = np.hstack((np.arange(len(diffs)), np.arange(len(diffs))))
        cols = np.hstack(([url_idx[url1] for (url1, url2) in diffs],
                          [url_idx[url2] for (url1, url2) in diffs]))
        # diffs are p2 - p1, so we want a positive value on the translation for p1,
        # e.g., a solution could be Tp1 == p2 - p1.
        Mvals = np.hstack(([pts.shape[1] for pts in diffs.values()],
                          [-pts.shape[1] for pts in diffs.values()]))
        print("solving")
        M = spp.csr_matrix((Mvals, (rows, cols)))

        # We use the sum of match differences
        D = np.vstack([d.sum(axis=1) for d in diffs.values()])
        oTx = lsqr(M, D[:, :1], damp=damping)[0]
        oTy = lsqr(M, D[:, 1:], damp=damping)[0]
        for k, idx in url_idx.iteritems():
            T[k][0] += oTx[idx]
            T[k][1] += oTy[idx]

        # first iteration is translation only
        if iter == 0:
            continue

        # don't update Rotations on last iteration
        if stepsize < 1e-30:
            print("Step size is small enough, finishing optimization")
            break

        # don't update Rotations on last iteration
        if (iter < maxiter - 1):
            # find points and their matches from other groups for each tile
            self_points = defaultdict(list)
            other_points = defaultdict(list)
            for (k1, k2), (p1, p2) in masked_matches.iteritems():
                self_points[k1].append(p1)
                self_points[k2].append(p2)
                other_points[k1].append(p2)
                other_points[k2].append(p1)
            self_points = {k: np.hstack(p) for k, p in self_points.iteritems()}
            other_points = {k: np.hstack(p) for k, p in other_points.iteritems()}

            self_centers = {k: np.mean(p, axis=1).reshape((2, 1)) for k, p in self_points.iteritems()}
            other_centers = {k: np.mean(p, axis=1).reshape((2, 1)) for k, p in other_points.iteritems()}

            # find best rotation, multiply the angle of rotation by a stepsize, and update the rotations
            new_R = {k: find_rotation(self_points[k] - self_centers[k],
                                      other_points[k] - other_centers[k],
                                      stepsize)
                     for k in self_centers}
            R = {k: np.dot(R[k], new_R[k]) for k in R}

    R = {k: v.tolist() for k, v in R.iteritems()}
    T = {k: v.tolist() for k, v in T.iteritems()}
    centers = {k: v.tolist() for k, v in centers.iteritems()}
    # json.dump({"Rotations": R,
    #            "Translations": T,
    #            "centers": centers},
    #           open(sys.argv[2], "wb"),
    #           indent=4)
    create_new_tilespec(tiles_fname, R, T, centers, out_fname)

if __name__ == '__main__':
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over a directory that contains matched points in json files, \
        optimizes these matches into a per-tile transformation, and saves a tilespec json file with these transformations.')
    parser.add_argument('tiles_fname', metavar='tiles_json', type=str,
                        help='a tile_spec file that contains the images to create sift features for, in json format')
    parser.add_argument('match_files_list', metavar='match_files_list', type=str,
                        help="a txt file containg a list of all the match files")
    parser.add_argument('-o', '--output_file', type=str,
                        help='an output tile_spec file, that will include the rotations for all tiles (default: ./output.json)',
                        default='./output.json')
    parser.add_argument('-c', '--conf_file_name', type=str,
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)

    args = parser.parse_args()

    optimize_2d_mfovs(args.tiles_fname, args.match_files_list, args.output_file, conf_fname=args.conf_file_name)
