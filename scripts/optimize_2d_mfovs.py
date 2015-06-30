import sys
import os.path
import os

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

if __name__ == "__main__":
    # all matched pairs between point sets
    all_matches = {}
    # all points from a given tile
    all_pts = defaultdict(list)

    match_files = glob.glob(os.path.join(sys.argv[1], '*sift_matches*.json'))
    print match_files
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

    # Find centers of each group of points
    centers = {k: np.mean(np.hstack(pts), axis=1, keepdims=True) for k, pts in all_pts.iteritems()}
    # a unique index for each url
    url_idx = {url: idx for idx, url in enumerate(all_pts)}

    maxiter = 1000
    epsilon = 5
    stepsize = 0.1
    prev_meanmed = np.inf
    damping = 0.01  # in units of matches per pair

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

    R = {k:v.tolist() for k, v in R.iteritems()}
    T = {k:v.tolist() for k, v in T.iteritems()}
    centers = {k:v.tolist() for k, v in centers.iteritems()}
    json.dump({"Rotations": R,
               "Translations": T,
               "centers": centers},
              open(sys.argv[2], "wb"),
              indent=4)
