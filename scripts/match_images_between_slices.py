# Setup
import PMCC_filter_example
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
# os.chdir("/data/SCS_2015-4-27_C1w7_alignment")
# os.chdir("/data/jpeg2k_test_sections_alignment")
datadir, imgdir, workdir, outdir = os.getcwd(), os.getcwd(), os.getcwd(), os.getcwd()


def analyzeimg(slicenumber, mfovnumber, num, data):
    slicestring = ("%03d" % slicenumber)
    numstring = ("%03d" % num)
    mfovstring = ("%06d" % mfovnumber)
    os.chdir(datadir)
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
    if (pointx > data[jsonindex]["bbox"][0] and pointy > data[jsonindex]["bbox"][2] and
            pointx < data[jsonindex]["bbox"][1] and pointy < data[jsonindex]["bbox"][3]):
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


def gettransformationbetween(slice1, mfov1, mfovmatches, data1):
    for i in range(0, len(mfovmatches["matches"])):
        if mfovmatches["matches"][i]["mfov1"] == mfov1:
            return mfovmatches["matches"][i]["transformation"]["matrix"]
    # Need to find a more intelligent way to do this, but this suffices for now
    # Uses transformation of another mfov (should be changed to the closest, unvisited mfov)
    # return gettransformationbetween(mfov1 - 1, mfovmatches)
    distances = []
    mfov1center = getcenter(slice1, mfov1, data1)
    for i in range(0, len(mfovmatches["matches"])):
        mfovma = mfovmatches["matches"][i]["mfov1"]
        mfovmacenter = getcenter(slice1, mfovma, data1)
        distances.append(distance.euclidean(mfov1center, mfovmacenter))
    closestindex = np.array(distances).argsort()[0]
    return mfovmatches["matches"][closestindex]["transformation"]["matrix"]


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
    checkindices = np.array(distances).argsort()[0]
    return indmatches[checkindices]


def getnumsfromindex(ind):
    return (ind / 61 + 1, ind % 61 + 1)


def getindexfromnums((mfovnum, imgnum)):
    return (mfovnum - 1) * 61 + imgnum - 1


def getimgmatches(slice1, slice2, nummfovs1, nummfovs2, data1, data2, mfovmatches):
    allimgsin1 = getimgcentersfromjson(data1)
    allimgsin2 = getimgcentersfromjson(data2)
    imgmatches = []

    for mfovnum in range(1, nummfovs1 + 1):
        for imgnum in range(1, 62):
            jsonindex = (mfovnum - 1) * 61 + imgnum - 1
            img1center = allimgsin1[jsonindex]
            expectedtransform = gettransformationbetween(slice1, mfovnum, mfovmatches, data1)
            expectednewcenter = np.dot(expectedtransform, np.append(img1center, [1]))[0:2]
            distances = np.zeros(len(allimgsin2))
            for j in range(0, len(allimgsin2)):
                distances[j] = np.linalg.norm(expectednewcenter - allimgsin2[j])
            checkindices = np.array(distances).argsort()[0:10]
            imgmatches.append((jsonindex, checkindices))
    return imgmatches


def getimgsfrominds(imginds, slice):
    imgarr = []
    for i in range(0, len(imginds)):
        slicestring = ("%03d" % slice)
        (imgmfov, imgnum) = getnumsfromindex(imginds[i])
        mfovstring = ("%06d" % imgmfov)
        numstring = ("%03d" % imgnum)
        imgurl = imgdir + slicestring + "/" + mfovstring + "/" + slicestring + "_" + mfovstring + "_" + numstring
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
            imgurl = imgdir + slicestring + "/" + mfovstring + "/" + slicestring + "_" + mfovstring + "_" + numstring
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


def gettemplatefromimgandpoint(img1resized, templatesize, centerpoint):
    imgheight = img1resized.shape[1]
    imgwidth = img1resized.shape[0]
    notonmesh = False

    xstart = centerpoint[0] - templatesize / 2
    ystart = centerpoint[1] - templatesize / 2
    xend = centerpoint[0] + templatesize / 2
    yend = centerpoint[1] + templatesize / 2

    if (xstart < 0):
        xend = 1 + xstart
        xstart = 1
        notonmesh = True
    if (ystart < 0):
        yend = 1 + ystart
        ystart = 1
        notonmesh = True
    if (xend >= imgwidth):
        diff = xend - imgwidth
        xstart -= diff + 1
        xend -= diff + 1
        notonmesh = True
    if (yend >= imgwidth):
        diff = yend - imgwidth
        ystart -= diff + 1
        yend -= diff + 1
        notonmesh = True

    if (xstart < 0) or (ystart < 0) or (xend >= imgwidth) or (yend >= imgheight):
        return None
    return (img1resized[xstart:(xstart + templatesize), ystart:(ystart + templatesize)].copy(), xstart, ystart, notonmesh)


def generatehexagonalgrid(boundingbox, spacing):
    hexheight = spacing
    hexwidth = math.sqrt(3) * spacing / 2
    vertspacing = 0.75 * hexheight
    horizspacing = hexwidth
    sizex = int((boundingbox[1] - boundingbox[0]) / horizspacing) + 2
    sizey = int((boundingbox[3] - boundingbox[2]) / vertspacing) + 2
    if sizey % 2 == 0:
        sizey += 1
    pointsret = []
    for i in range(-2, sizex):
        for j in range(-2, sizey):
            xpos = i * spacing
            ypos = j * spacing
            if j % 2 == 1:
                xpos += spacing * 0.5
            if (j % 2 == 1) and (i == sizex - 1):
                continue
            pointsret.append([int(xpos), int(ypos)])
    return pointsret


def findindwithinmatches(imgmatches, img1ind):
    for i in range(0, len(imgmatches)):
        (tempind1, __) = imgmatches[i]
        if tempind1 == img1ind:
            return i
    return -1

# <codecell>


def main():
    global datadir
    global imgdir
    global workdir
    global outdir
    script, slice1, slice2, datadir, imgdir, workdir, outdir, conffile = sys.argv
    slice1 = int(slice1)
    slice2 = int(slice2)
    slicestring1 = ("%03d" % slice1)
    slicestring2 = ("%03d" % slice2)

    os.chdir(datadir)
    with open("tilespecs/W01_Sec" + slicestring1 + ".json") as data_file1:
        data1 = json.load(data_file1)
    with open("tilespecs/W01_Sec" + slicestring2 + ".json") as data_file2:
        data2 = json.load(data_file2)
    with open(conffile) as conf_file:
        conf = json.load(conf_file)

    os.chdir(workdir)
    with open("Slice" + str(slice1) + "vs" + str(slice2) + ".json") as data_matches:
        mfovmatches = json.load(data_matches)
    nummfovs1 = len(data1) / 61
    nummfovs2 = len(data2) / 61

    if len(mfovmatches["matches"]) == 0:
        os.chdir(outdir)
        jsonfile = {}
        jsonfile['tilespec1'] = "file://" + os.getcwd() + "/tilespecs/W01_Sec" + ("%03d" % slice1) + ".json"
        jsonfile['tilespec2'] = "file://" + os.getcwd() + "/tilespecs/W01_Sec" + ("%03d" % slice2) + ".json"
        jsonfile['runtime'] = 0
        bb = getboundingbox(range(0, len(data1)), data1)
        hexgr = generatehexagonalgrid(bb, conf["template_matching_args"]["hexspacing"])
        jsonfile['mesh'] = hexgr
        finalpointmatches = []
        jsonfile['pointmatches'] = finalpointmatches
        json.dump(jsonfile, open("Images_Slice" + str(slice1) + "vs" + str(slice2) + ".json", 'w'), indent=4)
        return

    starttime = time.clock()
    imgmatches = getimgmatches(slice1, slice2, nummfovs1, nummfovs2, data1, data2, mfovmatches)

    bb = getboundingbox(range(0, len(data1)), data1)
    hexgr = generatehexagonalgrid(bb, conf["template_matching_args"]["hexspacing"])

    pointmatches = []
    scaling = conf["template_matching_args"]["scaling"]
    templatesize = conf["template_matching_args"]["templatesize"]

    for i in range(0, len(hexgr)):
        if i % 1000 == 0 and i > 0:
            print i
        img1ind = getclosestindtopoint(hexgr[i], slice1, data1)
        if img1ind is None:
            continue

        (img1ind, img2inds) = imgmatches[findindwithinmatches(imgmatches, img1ind)]
        (img1mfov, img1num) = getnumsfromindex(img1ind)
        slice1string = ("%03d" % slice1)
        mfov1string = ("%06d" % img1mfov)
        num1string = ("%03d" % img1num)
        img1url = imgdir + slice1string + "/" + mfov1string + "/" + slice1string + "_" + mfov1string + "_" + num1string

        img1 = cv2.imread(glob.glob(img1url + "*.bmp")[0], 0)
        img1resized = cv2.resize(img1, (0, 0), fx=scaling, fy=scaling)
        imgoffset1 = getimagetransform(slice1, img1mfov, img1num, data1)
        expectedtransform = gettransformationbetween(slice1, img1mfov, mfovmatches, data1)

        img1templates = gettemplatefromimgandpoint(img1resized, templatesize, (np.array(hexgr[i]) - imgoffset1) * scaling)
        if img1templates is None:
            continue

        chosentemplate, startx, starty, notonmesh = img1templates
        w, h = chosentemplate.shape[0], chosentemplate.shape[1]
        centerpoint1 = np.array([startx + w / 2, starty + h / 2]) / scaling + imgoffset1
        expectednewcenter = np.dot(expectedtransform, np.append(centerpoint1, [1]))[0:2]
        img2s = getimgsfromindsandpoint(img2inds, slice2, expectednewcenter, data2)
        ro, col = chosentemplate.shape
        rad2deg = -180 / math.pi
        angleofrot = rad2deg * math.atan2(expectedtransform[1][0], expectedtransform[0][0])
        rotationmatrix = cv2.getRotationMatrix2D((h / 2, w / 2), angleofrot, 1)
        rotatedtemp1 = cv2.warpAffine(chosentemplate, rotationmatrix, (col, ro))
        xaa = int(w / 2.9)
        rotatedandcroppedtemp1 = rotatedtemp1[(w / 2 - xaa):(w / 2 + xaa), (h / 2 - xaa):(h / 2 + xaa)]
        neww, newh = rotatedandcroppedtemp1.shape[0], rotatedandcroppedtemp1.shape[1]

        for k in range(0, len(img2s)):
            img2, img2ind = img2s[k]
            (img2mfov, img2num) = getnumsfromindex(img2ind)
            img2resized = cv2.resize(img2, (0, 0), fx=scaling, fy=scaling)
            imgoffset2 = getimagetransform(slice2, img2mfov, img2num, data2)

            # template1topleft = np.array([startx, starty]) / scaling + imgoffset1
            minco = conf["PMCC_args"]["min_correlation"]
            maxcu = conf["PMCC_args"]["maximal_curvature_ratio"]
            maxro = conf["PMCC_args"]["maximal_ROD"]
            result, reason = PMCC_filter_example.PMCC_match(img2resized, rotatedandcroppedtemp1, min_correlation=minco, maximal_curvature_ratio=maxcu, maximal_ROD=maxro)
            if result is not None:
                reasonx, reasony = reason
                # img1topleft = np.array([startx, starty]) / scaling + imgoffset1
                # img2topleft = np.array(reason) / scaling + imgoffset2
                img1centerpoint = np.array([startx + w / 2, starty + h / w]) / scaling + imgoffset1
                img2centerpoint = np.array([reasonx + neww / 2, reasony + newh / 2]) / scaling + imgoffset2
                pointmatches.append((img1centerpoint, img2centerpoint, notonmesh))

    os.chdir(outdir)
    jsonfile = {}
    jsonfile['tilespec1'] = "file://" + os.getcwd() + "/tilespecs/W01_Sec" + ("%03d" % slice1) + ".json"
    jsonfile['tilespec2'] = "file://" + os.getcwd() + "/tilespecs/W01_Sec" + ("%03d" % slice2) + ".json"
    jsonfile['runtime'] = time.clock() - starttime
    jsonfile['mesh'] = hexgr

    finalpointmatches = []
    for i in range(0, len(pointmatches)):
        p1, p2, nmesh = pointmatches[i]
        record = {}
        record['point1'] = p1.tolist()
        record['point2'] = p2.tolist()
        record['isvirtualpoint'] = nmesh
        finalpointmatches.append(record)

    jsonfile['pointmatches'] = finalpointmatches
    json.dump(jsonfile, open("Images_Slice" + str(slice1) + "vs" + str(slice2) + ".json", 'w'), indent=4)


if __name__ == '__main__':
    main()
