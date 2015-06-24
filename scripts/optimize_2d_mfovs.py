import sys
import os.path
import os

from collections import defaultdict
import json

import glob
import progressbar
import numpy as np

def dist(p1, p2):
    return np.sqrt(((p1 - p2) ** 2).sum(axis=0))

def find_rotation(p1, p2, scale):
    U, S, VT = np.linalg.svd(np.dot(p1, p2.T))
    print S
    R = np.dot(VT.T, U.T)
    angle = stepsize * np.arctan2(R[1, 0], R[0, 0])
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle),  np.cos(angle)]])

if __name__ == "__main__":
    # all matched pairs between point sets
    all_matches = {}
    # all points from a given tile
    all_pts = defaultdict(list)

    match_files = glob.glob(os.path.join(sys.argv[1], '*sift_matches*000004*000004*.json'))
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

    maxiter = 2000
    epsilon = 5
    stepsize = 0.01
    prev_meanmed = np.inf

    T = defaultdict(lambda: np.zeros((2, 1)))
    R = defaultdict(lambda: np.eye(2))
    for iter in range(maxiter):
        if stepsize < 1e-30:
            break

        # transform points by the current trans/rot
        trans_matches = {(k1, k2): (np.dot(R[k1], p1) + T[k1],
                                    np.dot(R[k2], p2) + T[k2])
                         for (k1, k2), (p1, p2) in all_matches.iteritems()}

        # mask off all points more than epsilon past the median
        distances = {k: dist(p1, p2) for k, (p1, p2) in trans_matches.iteritems()}
        masks = {k : d < (np.median(d) + epsilon) for k, d in distances.iteritems()}
        masked_matches = {k: (p1[:, masks[k]], p2[:, masks[k]]) for k, (p1, p2) in trans_matches.iteritems()}

        median_dists = [np.median(d) for d in distances.values()]
        medmed = np.median(median_dists)
        meanmed = np.mean(median_dists)
        print("med-med distance: {}, mean-med distance: {}  SZ: {}".format(medmed, meanmed, stepsize))
        if meanmed < prev_meanmed:
            stepsize *= 1.1
            if stepsize > 1:
                stepsize = 1
        else:
            stepsize *= 0.5
        prev_meanmed = meanmed

        # find points and their matches from other groups for each tile
        self_points = defaultdict(list)
        other_points = defaultdict(list)
        for (k1, k2), (p1, p2) in masked_matches.iteritems():
            self_points[k1].append(p1)
            self_points[k2].append(p2)
            other_points[k1].append(p2)
            other_points[k2].append(p1)
        self_points = {k : np.hstack(p) for k, p in self_points.iteritems()}
        other_points = {k : np.hstack(p) for k, p in other_points.iteritems()}

        # update translations
        self_centers = {k : np.mean(p, axis=1).reshape((2, 1)) for k, p in self_points.iteritems()}
        other_centers = {k : np.mean(p, axis=1).reshape((2, 1)) for k, p in other_points.iteritems()}

        # find best rotation, multiply the angle of rotation by a stepsize, and update the rotations
        new_R = {k: find_rotation(self_points[k] - self_centers[k],
                                  other_points[k] - other_centers[k],
                                  stepsize)
                 for k in self_centers}
        R = {k: np.dot(R[k], new_R[k]) for k in R}
        deltas = {k: stepsize * (other_centers[k] - np.dot(new_R[k], self_centers[k])) for k in self_centers}
        T = {k: old_translation + deltas[k] for k, old_translation in T.iteritems()}
