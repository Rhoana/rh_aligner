# Setup
import utils
import models
import ransac
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import random
import math
import sys
import getopt
import operator
from scipy.spatial import Delaunay
from scipy.spatial import distance
from scipy.spatial import KDTree
import cv2
import time
import glob
os.chdir("C:/Users/Raahil/Documents/Research2015_eclipse/Testing")
# os.chdir("/data/SCS_2015-4-27_C1w7_alignment")


def analyzeimg(slicenumber, mfovnumber, num, data):
    slicestring = ("%03d" % slicenumber)
    numstring = ("%03d" % num)
    mfovstring = ("%06d" % mfovnumber)
    imgname = "2d_work_dir/W01_Sec" + slicestring + "/W01_Sec" + slicestring + "_sifts_" + slicestring + "_" + mfovstring + "_" + numstring + "*"
    f = h5py.File(glob.glob(imgname)[0], 'r')
    resps = f['pts']['responses'][:]
    descs = f['descs'][:]
    octas = f['pts']['octaves'][:]
    jsonindex = (mfovnumber - 1) * 61 + num - 1
    xtransform = float(data[jsonindex]["transforms"][0]["dataString"].encode("ascii").split(" ")[0])
    ytransform = float(data[jsonindex]["transforms"][0]["dataString"].encode("ascii").split(" ")[1])

    xlocs = []
    ylocs = []
    if len(resps) != 0:
        xlocs = f['pts']['locations'][:, 0] + xtransform
        ylocs = f['pts']['locations'][:, 1] + ytransform

    allpoints = []
    allresps = []
    alldescs = []
    for pointindex in range(0, len(xlocs)):
        currentocta = int(octas[pointindex]) & 255
        if currentocta > 128:
            currentocta -= 255
        if currentocta == 4 or currentocta == 5:
            allpoints.append(np.array([xlocs[pointindex], ylocs[pointindex]]))
            allresps.append(resps[pointindex])
            alldescs.append(descs[pointindex])
    points = np.array(allpoints).reshape((len(allpoints), 2))
    return (points, allresps, alldescs)


def getimagecenter(slicenumber, mfovnumber, imgnumber, data):
    xlocsum, ylocsum, nump = 0, 0, 0
    jsonindex = (mfovnumber - 1) * 61 + imgnumber - 1
    xlocsum += data[jsonindex]["bbox"][0] + data[jsonindex]["bbox"][1]
    ylocsum += data[jsonindex]["bbox"][2] + data[jsonindex]["bbox"][3]
    nump += 2
    return [xlocsum / nump, ylocsum / nump]

    
def getcenter(slicenumber, mfovnumber, data):
    xlocsum, ylocsum, nump = 0, 0, 0
    for num in range(1, 62):
        jsonindex = (mfovnumber - 1) * 61 + num - 1
        xlocsum += data[jsonindex]["bbox"][0] + data[jsonindex]["bbox"][1]
        ylocsum += data[jsonindex]["bbox"][2] + data[jsonindex]["bbox"][3]
        nump += 2
    return [xlocsum / nump, ylocsum / nump]


def generatematches_cv2(allpoints1, allpoints2, alldescs1, alldescs2):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(np.array(alldescs1), np.array(alldescs2), k=2)
    goodmatches = []
    for m, n in matches:
        if m.distance / n.distance < 0.92:
            goodmatches.append([m])
    match_points = np.array([
        np.array([allpoints1[[m[0].queryIdx for m in goodmatches]]][0]),
        np.array([allpoints2[[m[0].trainIdx for m in goodmatches]]][0])])
    return match_points


def generatematches_brute(allpoints1, allpoints2, alldescs1, alldescs2):
    bestpoints1 = []
    bestpoints2 = []
    for pointrange in range(0, len(allpoints1)):
        selectedpoint = allpoints1[pointrange]
        selectedpointd = alldescs1[pointrange]
        bestdistsofar = sys.float_info.max - 1
        secondbestdistsofar = sys.float_info.max
        bestcomparedpoint = allpoints2[0]
        distances = []
        for num in range(0, len(allpoints2)):
            comparedpointd = alldescs2[num]
            bestdist = distance.euclidean(selectedpointd.astype(np.int), comparedpointd.astype(np.int))
            distances.append(bestdist)
            if bestdist < bestdistsofar:
                secondbestdistsofar = bestdistsofar
                bestdistsofar = bestdist
                bestcomparedpoint = allpoints2[num]
            elif bestdist < secondbestdistsofar:
                secondbestdistsofar = bestdist
        if bestdistsofar / secondbestdistsofar < .92:
            bestpoints1.append(selectedpoint)
            bestpoints2.append(bestcomparedpoint)
    match_points = np.array([bestpoints1, bestpoints2])
    return match_points


def gettransformationbetween(mfov1, mfovmatches):
    for i in range(0, len(mfovmatches["matches"])):
        if mfovmatches["matches"][i]["mfov1"] == mfov1:
            return mfovmatches["matches"][i]["transformation"]["matrix"]
    # Need to find a more intelligent way to do this, but this suffices for now
    # Uses transformation of another mfov (should be changed to the closest, unvisited mfov)
    return gettransformationbetween(mfov1 - 1, mfovmatches)


def getimgcentersfromjson(data):
    centers = []
    for i in range(0, len(data)):
        xlocsum, ylocsum, nump = 0
        xlocsum += data[i]["bbox"][0] + data[i]["bbox"][1]
        ylocsum += data[i]["bbox"][2] + data[i]["bbox"][3]
        nump += 2
        centers.append([xlocsum / nump, ylocsum / nump])
    return centers

slice1 = 90
slice2 = 91
nummfovs = 5
slicestring1 = ("%03d" % slice1)
slicestring2 = ("%03d" % slice2)
with open("tilespecs/W01_Sec" + slicestring1 + ".json") as data_file1:
    data1 = json.load(data_file1)
with open("tilespecs/W01_Sec" + slicestring2 + ".json") as data_file2:
    data2 = json.load(data_file2)
with open("Slice" + str(slice1) + "vs" + str(slice2) + ".json") as data_matches:
    mfovmatches = json.load(data_matches)
    
# jsonindex = (mfovnumber - 1) * 61 + imgnumber - 1
