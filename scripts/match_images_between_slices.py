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


def getclosestindtopoint(point, slicenumber, data):
    indmatches = getimgindsfrompoint(point, slicenumber, data)
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


def getimgmatches(slice1, slice2, nummfovs):
    slicestring1 = ("%03d" % slice1)
    slicestring2 = ("%03d" % slice2)
    with open("tilespecs/W01_Sec" + slicestring1 + ".json") as data_file1:
        data1 = json.load(data_file1)
    with open("tilespecs/W01_Sec" + slicestring2 + ".json") as data_file2:
        data2 = json.load(data_file2)
    with open("/home/raahilsha/Slice" + str(slice1) + "vs" + str(slice2) + ".json") as data_matches:
        mfovmatches = json.load(data_matches)
        
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
    numslices = int(imgwidth / templatesize) - 1
    searchsq = templatesize
    for i in range(0, numslices):
        for j in range(0, numslices):
            ystart = imgheight / numslices * i
            xstart = imgwidth / numslices * j
            template = img1[ystart:(ystart + searchsq), xstart:(xstart + searchsq)].copy()
            templates.append((template, xstart, ystart))
    return templates

# <codecell>

slice1 = 2
slice2 = 3
nummfovs = 53
imgmatches = getimgmatches(slice1, slice2, nummfovs)

# <codecell>

slicestring1 = ("%03d" % slice1)
slicestring2 = ("%03d" % slice2)
with open("tilespecs/W01_Sec" + slicestring1 + ".json") as data_file1:
    data1 = json.load(data_file1)
with open("tilespecs/W01_Sec" + slicestring2 + ".json") as data_file2:
    data2 = json.load(data_file2)
with open("/home/raahilsha/Slice" + str(slice1) + "vs" + str(slice2) + ".json") as data_matches:
    mfovmatches = json.load(data_matches)

# <codecell>

pointmatches = []
starttime = time.clock()
scaling = 0.2
templatesize = 200

for i in range(0, len(imgmatches)):
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
    
    for j in range(0, len(img1templates)):
        chosentemplate, startx, starty = img1templates[j]
        w, h = chosentemplate.shape[0], chosentemplate.shape[1]
        centerpoint1 = np.array([startx + w / 2, starty + h / 2]) / scaling + imgoffset1
        expectednewcenter = np.dot(expectedtransform, np.append(centerpoint1, [1]))[0:2]
        img2s = getimgsfromindsandpoint(img2inds, slice2, expectednewcenter, data2)
        for k in range(0, len(img2s)):
            img2, img2ind = img2s[k]
            (img2mfov, img2num) = getnumsfromindex(img2ind)
            img2resized = cv2.resize(img2, (0, 0), fx = scaling, fy = scaling)
            imgoffset2 = getimagetransform(slice2, img2mfov, img2num, data2)
            
            template1topleft = centerpoint1 = np.array([startx, starty]) / scaling + imgoffset1
            result, reason = PMCC_filter_example.PMCC_match(img2resized, chosentemplate, min_correlation=0.2)
            if result is not None:            
                img1topleft = np.array([startx, starty]) / scaling + imgoffset1
                img2topleft = np.array(reason) / scaling + imgoffset2
                pointmatches.append((img1topleft, img2topleft))
            
            '''
            res = cv2.matchTemplate(img2resized, chosentemplate, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            centerpoint2 = np.array([top_left[0] + w / 2, top_left[1] + h / 2]) / scaling + imgoffset2
            
            pointmatches.append((centerpoint1, centerpoint2))
            '''

# <codecell>

(img1ind, img2inds) = random.choice(imgmatches)
(img1mfov, img1num) = getnumsfromindex(img1ind)

slice1string = ("%03d" % slice1)
mfov1string = ("%06d" % img1mfov)
num1string = ("%03d" % img1num)
img1url = "/data/images/SCS_2015-4-27_C1w7/" + slice1string + "/" + mfov1string + "/" + slice1string + "_" + mfov1string + "_" + num1string

img1 = cv2.imread(glob.glob(img1url + "*.bmp")[0], 0)
img2s = getimgsfrominds(img2inds, slice2)
img1templates = gettemplatesfromimg(img1)

# Later on, we can use scipy's ndimage to rotate the templates like this:
# rotate_lena_noreshape = ndimage.rotate(lena, 45, reshape=False

randtemplate, randstartx, randstarty = random.choice(img1templates)
img2 = img2s[0].copy()
(img2mfov, img2num) = getnumsfromindex(img2inds[0])

w = randtemplate.shape[0]
h = randtemplate.shape[1]
meth = "cv2.TM_CCOEFF_NORMED"
method = eval(meth)
res = cv2.matchTemplate(img2, randtemplate, method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

slicestring1 = ("%03d" % slice1)
slicestring2 = ("%03d" % slice2)
with open("tilespecs/W01_Sec" + slicestring1 + ".json") as data_file1:
    data1 = json.load(data_file1)
with open("tilespecs/W01_Sec" + slicestring2 + ".json") as data_file2:
    data2 = json.load(data_file2)
with open("/home/raahilsha/Slice" + str(slice1) + "vs" + str(slice2) + ".json") as data_matches:
    mfovmatches = json.load(data_matches)
imgoffset1 = getimagetransform(slice1, img1mfov, img1num, data1)
imgoffset2 = getimagetransform(slice2, img2mfov, img2num, data2)
expectedtransform = gettransformationbetween(img1mfov, mfovmatches)
centerpoint1 = np.array([randstartx + w / 2, randstarty + h / 2]) + imgoffset1
expectednewcenter = np.dot(expectedtransform, np.append(centerpoint1, [1]))[0:2] - imgoffset2
centerpoint1 = centerpoint1 - imgoffset1

cv2.rectangle(img2,top_left, bottom_right, 255, 20)
cv2.circle(img2, (int(expectednewcenter[0]), int(expectednewcenter[1])), 20, 255, -1)
plt1fig = img1.copy()
cv2.rectangle(plt1fig, (randstartx, randstarty), (randstartx + w, randstarty + h), 255, 20)
cv2.circle(plt1fig, (int(centerpoint1[0]), int(centerpoint1[1])), 20, 255, -1)

%matplotlib
plt.figure(1)
plt.imshow(plt1fig)
plt.figure(2)
plt.imshow(img2)
plt.figure(3)
plt.imshow(res)

# <codecell>

# Use img1 and img2
scalings = [.2, .3, .5, .75, 1]
templatesizes = [5, 10, 50, 75, 100, 125, 150, 175, 200, 300, 400]
finalarr = np.zeros((len(scalings), len(templatesizes)))

for scalingr in range(0, len(scalings)):
    scaling = scalings[scalingr]
    print scaling
    for templater in range(0, len(templatesizes)):
        templatesize = templatesizes[templater]
        timearr = []
        img1resized = cv2.resize(img1, (0, 0), fx = scaling, fy = scaling)
        img2resized = cv2.resize(img2, (0, 0), fx = scaling, fy = scaling)
        img1templates = gettemplatesfromimg(img1resized, templatesize)
        
        k = len(img1templates)
        if k > 3:
            k = 3
        for i in range(0, k):
            for boo in range(0,5):
                starttime = time.clock()
                method = cv2.TM_CCOEFF_NORMED
                chosentemplate, startx, starty = img1templates[i]
                res = cv2.matchTemplate(img2resized, chosentemplate, method)
                timearr.append(time.clock() - starttime)
        if len(timearr) != 0:
            avgtime = (sum(timearr) / len(timearr)) * (float(img1resized.size) / res.size)
        else:
            avgtime = 0
        finalarr[scalingr, templater] = avgtime

for row in finalarr:
    plt.plot(templatesizes, row)
plt.show()

# <codecell>

%matplotlib
plt.figure(1)
plt.imshow(img1resized)
plt.figure(2)
plt.imshow(img2resized)
plt.figure(3)
plt.imshow(chosentemplate)

# <codecell>

img1url

# <codecell>

img2s

# <codecell>

getnumsfromindex(314)

# <codecell>

np.array((10,10))

# <codecell>
