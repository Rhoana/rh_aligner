# Setup
import os
import numpy as np
import h5py
import json
import math
import sys
from scipy.spatial import distance
import cv2
import time
import glob
import matplotlib.pyplot as plt
from pylab import axis

def main():
    script, jsonfile = sys.argv
    with open(jsonfile) as data_file1:
            data1 = json.load(data_file1)
    
    pms = data1['pointmatches']
    pointmatches = []
    for i in range(0, len(pms)):
        pointmatches.append((np.array(pms[i]['point1']), np.array(pms[i]['point2'])))
    
    point1s = map(list, zip(*pointmatches))[0]
    point1s = map(lambda x: np.matrix(x).T, point1s)
    point2s = map(list, zip(*pointmatches))[1]
    point2s = map(lambda x: np.matrix(x).T, point2s)
    centroid1 = [np.array(point1s)[:,0].mean(), np.array(point1s)[:,1].mean()]
    centroid2 = [np.array(point2s)[:,0].mean(), np.array(point2s)[:,1].mean()]
    h = np.matrix(np.zeros((2,2)))
    for i in range(0, len(point1s)):
        sumpart = (np.matrix(point1s[i]) - centroid1).dot((np.matrix(point2s[i]) - centroid2).T)
        h = h + sumpart
    U, S, Vt = np.linalg.svd(h)
    R = Vt.T.dot(U.T)
    plt.figure(1)
    for i in range(0,len(pointmatches)):
        point1, point2 = pointmatches[i]
        point1 = np.matrix(point1 - centroid1).dot(R.T).tolist()[0]
        point2 = point2 - centroid2
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]])
        axis('equal')
    plt.show()

if __name__ == '__main__':
    main()
