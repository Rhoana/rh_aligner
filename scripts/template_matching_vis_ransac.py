# Setup
import ransac
import numpy as np
import json
import sys
import matplotlib.pyplot as plt
from matplotlib import collections as mc


def main(jsonfile):
    with open(jsonfile) as data_file1:
        data1 = json.load(data_file1)

    pms = data1['pointmatches']
    points1 = np.array([p['point1'] for p in pms])
    points2 = np.array([p['point2'] for p in pms])

    try:
        if points1.shape[1] != 2:
            return
    except:
        return

    centroid1 = points1.mean(axis=0, keepdims=True)
    centroid2 = points2.mean(axis=0, keepdims=True)

    model_index = 1
    iterations = 500
    max_epsilon = 100
    min_inlier_ratio = 0
    min_num_inlier = 7
    max_trust = 3
    pointmatchesr = (points1, points2)
    try:
        model, filtered_matches = ransac.filter_matches(pointmatchesr, model_index, iterations, max_epsilon, min_inlier_ratio, min_num_inlier, max_trust)
        R = model.get_matrix()[0:2, 0:2]
    except:
        return
    '''
    h = np.matrix(np.zeros((2,2)))
    for i in range(0, len(point1s)):
        sumpart = (np.matrix(point1s[i]) - centroid1).dot((np.matrix(point2s[i]) - centroid2).T)
        h = h + sumpart
    U, S, Vt = np.linalg.svd(h)
    R = Vt.T.dot(U.T)
    print R
    '''

    new_points1 = np.dot(R, (points1 - centroid1).T).T
    new_points2 = points2 - centroid2
    lines = [[p1, p2] for p1, p2 in zip(new_points1, new_points2)]
    lc = mc.LineCollection(lines)
    plt.figure()
    plt.gca().add_collection(lc)
    plt.scatter(new_points1[:, 0], new_points1[:, 1])
    plt.gca().autoscale()
    plt.axis('equal')
    plt.title(jsonfile)
    return model.get_matrix()

if __name__ == '__main__':
    transforms = [main(arg) for arg in sys.argv[1:]]
    if len(transforms) == 2:
        RT1, RT2 = transforms
        print "Composed transform, last column should be [0, 0, 1].T:\n", np.dot(RT1, RT2)
    plt.show()
