# Setup
import numpy as np
import json
import sys
import matplotlib
matplotlib.use('GTK')
import matplotlib.pyplot as plt
from pylab import axis
import utils
from bounding_box import BoundingBox


def main(jsonfile):
    with open(jsonfile, 'r') as data_file1:
        data1 = json.load(data_file1)

    ts1_fname = data1["tilespec1"].replace("file://", "")

    with open(ts1_fname, 'r') as ts_file1:
        ts1 = json.load(ts_file1)
        ts1_indexed = utils.index_tilespec(ts1)

    # The mfovs list in the match file must be a subset of the mfovs list in the tilespec
    mfov1_to_center2 = {}
    for mfov_match in data1["matches"]:
        mfov1_to_center2[mfov_match["mfov1"]] = np.array(mfov_match["section2_center"])
    centers2 = [mfov1_to_center2[mfov1] for mfov1 in sorted(mfov1_to_center2.keys())]

    mfovs1_bb = [BoundingBox.read_bbox_from_ts(ts1_indexed[mfov].values()) for mfov in sorted(mfov1_to_center2.keys())]
    centers1 = [np.array([(bb.from_x + bb.to_x) / 2.0, (bb.from_y + bb.to_y) / 2.0]) for bb in mfovs1_bb]

    pointmatches = []
    for i in range(0, len(centers1)):
        pointmatches.append((centers1[i], centers2[i]))
    if (len(pointmatches) == 0):
        return

    point1s = map(list, zip(*pointmatches))[0]
    point1s = map(lambda x: np.matrix(x).T, point1s)
    point2s = map(list, zip(*pointmatches))[1]
    point2s = map(lambda x: np.matrix(x).T, point2s)
    centroid1 = [np.array(point1s)[:, 0].mean(), np.array(point1s)[:, 1].mean()]
    centroid2 = [np.array(point2s)[:, 0].mean(), np.array(point2s)[:, 1].mean()]
    h = np.matrix(np.zeros((2, 2)))
    for i in range(0, len(point1s)):
        sumpart = (np.matrix(point1s[i]) - centroid1).dot((np.matrix(point2s[i]) - centroid2).T)
        h = h + sumpart
    U, S, Vt = np.linalg.svd(h)
    R = Vt.T.dot(U.T)
    plt.figure()
    for i in range(0, len(pointmatches)):
        point1, point2 = pointmatches[i]
        point1 = np.matrix(point1 - centroid1).dot(R.T).tolist()[0]
        point2 = point2 - centroid2
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]])
        plt.scatter(point1[0], point1[1])
        axis('equal')

if __name__ == '__main__':
    transforms = [main(arg) for arg in sys.argv[1:]]
    plt.show()
