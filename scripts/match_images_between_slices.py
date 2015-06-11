# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# Setup
import utils
import models
import ransac
import PMCC_filter_example
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
from scipy import ndimage
from pylab import axis
import cv2
import time
import glob
os.chdir("/data/SCS_2015-4-27_C1w7_alignment")


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


def getimagetransform(slicenumber, mfovnumber, num, data):
    jsonindex = (mfovnumber - 1) * 61 + num - 1
    xtransform = float(data[jsonindex]["transforms"][0]["dataString"].encode("ascii").split(" ")[0])
    ytransform = float(data[jsonindex]["transforms"][0]["dataString"].encode("ascii").split(" ")[1])
    return [xtransform, ytransform]


def getimagetopleft(slicenumber, mfovnumber, num, data):
    jsonindex = (mfovnumber - 1) * 61 + num - 1
    xlocsum = data[jsonindex]["bbox"][0]
    ylocsum = data[jsonindex]["bbox"][2]
    return [xlocsum, ylocsum]


def getimagebottomright(slicenumber, mfovnumber, num, data):
    jsonindex = (mfovnumber - 1) * 61 + num - 1
    xlocsum = data[jsonindex]["bbox"][1]
    ylocsum = data[jsonindex]["bbox"][3]
    return [xlocsum, ylocsum]


def imghittest(point, slicenumber, mfovnumber, imgnumber, data):
    pointx = point[0]
    pointy = point[1]
    jsonindex = (mfovnumber - 1) * 61 + imgnumber - 1
    if (pointx > data[jsonindex]["bbox"][0] and pointy > data[jsonindex]["bbox"][2] 
            and pointx < data[jsonindex]["bbox"][1] and pointy < data[jsonindex]["bbox"][3]):
        return True
    return False


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
        xlocsum, ylocsum, nump = 0, 0, 0
        xlocsum += data[i]["bbox"][0] + data[i]["bbox"][1]
        ylocsum += data[i]["bbox"][2] + data[i]["bbox"][3]
        nump += 2
        centers.append([xlocsum / nump, ylocsum / nump])
    return centers


def getimgindsfrompoint(point, slicenumber, data):
    indmatches = []
    for i in range(0, len(data)):
        (mfovnum, numnum) = getnumsfromindex(i)
        if imghittest(point, slicenumber, mfovnum, numnum, data):
            indmatches.append(i)
    return indmatches


def getboundingbox(imgindlist, data):
    finalbox = [data[0]["bbox"][0], data[0]["bbox"][1], data[0]["bbox"][2], data[0]["bbox"][3]]
    for i in range(0, len(data)):
        if data[i]["bbox"][0] < finalbox[0]:
            finalbox[0] = data[i]["bbox"][0]
        if data[i]["bbox"][1] > finalbox[1]:
            finalbox[1] = data[i]["bbox"][1]
        if data[i]["bbox"][2] < finalbox[2]:
            finalbox[2] = data[i]["bbox"][2]
        if data[i]["bbox"][3] > finalbox[3]:
            finalbox[3] = data[i]["bbox"][3]
    return finalbox


def getclosestindtopoint(point, slicenumber, data):
    indmatches = getimgindsfrompoint(point, slicenumber, data)
    if (len(indmatches) == 0):
        return None
    distances = []
    for i in range(0, len(indmatches)):
        (mfovnum, numnum) = getnumsfromindex(indmatches[i])
        center = getimagecenter(slicenumber, mfovnum, numnum, data)
        dist = np.linalg.norm(np.array(center) - np.array(point))
        distances.append(dist)
    checkindices = distances.argsort()[0]
    return checkindices


def getnumsfromindex(ind):
    return (ind / 61 + 1, ind % 61 + 1)


def getindexfromnums((mfovnum, imgnum)):
    return (mfovnum - 1) * 61 + imgnum - 1


def getimgmatches(slice1, slice2, nummfovs, data1, data2, mfovmatches):
    allimgsin1 = getimgcentersfromjson(data1)
    allimgsin2 = getimgcentersfromjson(data2)
    imgmatches = []
    
    for mfovnum in range(1, nummfovs + 1):
        for imgnum in range(1, 62):
            jsonindex = (mfovnum - 1) * 61 + imgnum - 1
            img1center = allimgsin1[jsonindex]
            expectedtransform = gettransformationbetween(mfovnum, mfovmatches)
            expectednewcenter = np.dot(expectedtransform, np.append(img1center, [1]))[0:2]
            distances = np.zeros(len(allimgsin2))
            for j in range(0, len(allimgsin2)):
                distances[j] = np.linalg.norm(expectednewcenter - allimgsin2[j])
            checkindices = distances.argsort()[0:10]
            imgmatches.append((jsonindex, checkindices))
    return imgmatches


def getimgsfrominds(imginds, slice):
    imgarr = []
    for i in range(0, len(imginds)):
        slicestring = ("%03d" % slice)
        (imgmfov, imgnum) = getnumsfromindex(imginds[i])
        mfovstring = ("%06d" % imgmfov)
        numstring = ("%03d" % imgnum)
        imgurl = "/data/images/SCS_2015-4-27_C1w7/" + slicestring + "/" + mfovstring + "/" + slicestring + "_" + mfovstring + "_" + numstring
        imgt = cv2.imread(glob.glob(imgurl + "*.bmp")[0], 0)
        imgarr.append(imgt)
    return imgarr


def getimgsfromindsandpoint(imginds, slicenumber, point, data):
    imgarr = []
    for i in range(0, len(imginds)):
        (imgmfov, imgnum) = getnumsfromindex(imginds[i])
        if imghittest(point, slicenumber, imgmfov, imgnum, data):
            slicestring = ("%03d" % slicenumber)
            mfovstring = ("%06d" % imgmfov)
            numstring = ("%03d" % imgnum)
            imgurl = "/data/images/SCS_2015-4-27_C1w7/" + slicestring + "/" + mfovstring + "/" + slicestring + "_" + mfovstring + "_" + numstring
            imgt = cv2.imread(glob.glob(imgurl + "*.bmp")[0], 0)
            imgarr.append((imgt, imginds[i]))
    return imgarr


def gettemplatesfromimg(img1, templatesize):
    imgheight = img1.shape[0]
    imgwidth = img1.shape[1]
    templates = []
    numslices = int(imgwidth / templatesize)
    searchsq = templatesize
    for i in range(0, numslices):
        for j in range(0, numslices):
            ystart = imgheight / numslices * i
            xstart = imgwidth / numslices * j
            template = img1[ystart:(ystart + searchsq), xstart:(xstart + searchsq)].copy()
            templates.append((template, xstart, ystart))
    return templates

def generatehexagonalgrid(boundingbox, spacing):
    sizex = int((boundingbox[1] - boundingbox[0]) / spacing)
    sizey = int((boundingbox[3] - boundingbox[2]) / spacing)
    pointsret = []
    for i in range(0, sizex):
        for j in range(0, sizey):
            xpos = i * spacing
            ypos = j * spacing
            if j % 2 == 0:
                xpos += spacing * 0.5
            pointsret.append([xpos, ypos])
    return pointsret

# <codecell>

slice1 = 2
slice2 = 3
nummfovs = 53
slicestring1 = ("%03d" % slice1)
slicestring2 = ("%03d" % slice2)
with open("tilespecs/W01_Sec" + slicestring1 + ".json") as data_file1:
    data1 = json.load(data_file1)
with open("tilespecs/W01_Sec" + slicestring2 + ".json") as data_file2:
    data2 = json.load(data_file2)
with open("/home/raahilsha/Slice" + str(slice1) + "vs" + str(slice2) + ".json") as data_matches:
    mfovmatches = json.load(data_matches)

# <codecell>

starttime = time.clock()
imgmatches = getimgmatches(slice1, slice2, nummfovs, data1, data2, mfovmatches)
print "Runtime: " + str(time.clock() - starttime) + " seconds"

# <codecell>

pointmatches = []
starttime = time.clock()
scaling = 0.2
templatesize = 300

for i in range(0, len(imgmatches)):
    if i % 100 == 0:
        print str(float(i) / len(imgmatches) * 100) + "% done"
    (img1ind, img2inds) = imgmatches[i]
    (img1mfov, img1num) = getnumsfromindex(img1ind)

    slice1string = ("%03d" % slice1)
    mfov1string = ("%06d" % img1mfov)
    num1string = ("%03d" % img1num)
    img1url = "/data/images/SCS_2015-4-27_C1w7/" + slice1string + "/" + mfov1string + "/" + slice1string + "_" + mfov1string + "_" + num1string

    img1 = cv2.imread(glob.glob(img1url + "*.bmp")[0], 0)
    img1resized = cv2.resize(img1, (0, 0), fx = scaling, fy = scaling)
    img1templates = gettemplatesfromimg(img1resized, templatesize)
    imgoffset1 = getimagetransform(slice1, img1mfov, img1num, data1)
    expectedtransform = gettransformationbetween(img1mfov, mfovmatches)
    # rotationmatrix = np.matrix(np.array(expectedtransform[0:2]).T[0:2].T)
    
    for j in range(0, len(img1templates)):
        chosentemplate, startx, starty = img1templates[j]
        w, h = chosentemplate.shape[0], chosentemplate.shape[1]
        centerpoint1 = np.array([startx + w / 2, starty + h / 2]) / scaling + imgoffset1
        expectednewcenter = np.dot(expectedtransform, np.append(centerpoint1, [1]))[0:2]
        img2s = getimgsfromindsandpoint(img2inds, slice2, expectednewcenter, data2)
        ro, col = chosentemplate.shape
        rad2deg = -180 / math.pi
        angleofrot = rad2deg * math.atan2(expectedtransform[1][0], expectedtransform[0][0])
        rotationmatrix = cv2.getRotationMatrix2D((h / 2, w / 2), angleofrot, 1)
        rotatedtemp1 = cv2.warpAffine(chosentemplate, rotationmatrix, (col, ro))
        xaa = int(w / 2.9) # Divide by a bit more than the square root of 2
        rotatedandcroppedtemp1 = rotatedtemp1[(w / 2 - xaa):(w / 2 + xaa), (h / 2 - xaa):(h / 2 + xaa)]
        neww, newh = rotatedandcroppedtemp1.shape[0], rotatedandcroppedtemp1.shape[1]
        
        for k in range(0, len(img2s)):
            img2, img2ind = img2s[k]
            (img2mfov, img2num) = getnumsfromindex(img2ind)
            img2resized = cv2.resize(img2, (0, 0), fx = scaling, fy = scaling)
            imgoffset2 = getimagetransform(slice2, img2mfov, img2num, data2)
            
            template1topleft = np.array([startx, starty]) / scaling + imgoffset1
            result, reason = PMCC_filter_example.PMCC_match(img2resized, rotatedandcroppedtemp1, min_correlation=0.3)
            if result is not None:
                reasonx, reasony = reason
                img1topleft = np.array([startx, starty]) / scaling + imgoffset1
                img2topleft = np.array(reason) / scaling + imgoffset2
                img1centerpoint = np.array([startx + w / 2, starty + h / w]) / scaling + imgoffset1
                img2centerpoint = np.array([reasonx + neww / 2, reasony + newh / 2]) / scaling + imgoffset2
                pointmatches.append((img1centerpoint, img2centerpoint))
            
print "Runtime: " + str(time.clock() - starttime) + " seconds"

# <codecell>

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

# <codecell>

%matplotlib
plt.figure(1)
for i in range(0,len(pointmatches)):
    point1, point2 = pointmatches[i]
    # point1 = np.matrix(point1 - centroid1).dot(R.T).tolist()[0]
    # point2 = point2 - centroid2
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]])
    axis('equal')

# <codecell>

bb = getboundingbox(range(0, len(data1)), data1)
hexgr = generatehexagonalgrid(bb, 1500)
%matplotlib
plt.figure(1)
for i in range(0, len(hexgr)):
    if i % 1000 == 0:
        print i
    plt.scatter(hexgr[i][0], hexgr[i][1])
    axis('equal')

# <codecell>

bb = getboundingbox(range(0, len(data1)), data1)
hexgr = generatehexagonalgrid(bb, 1500)
len(hexgr)

# <codecell>


