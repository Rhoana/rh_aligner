# Setup
import models
import ransac
import os
import numpy as np
import h5py
import json
import random
import sys
from scipy.spatial import distance
import cv2
import time
import glob
# os.chdir("C:/Users/Raahil/Documents/Research2015_eclipse/Testing")
# os.chdir("/data/SCS_2015-4-27_C1w7_alignment")
# os.chdir("/data/jpeg2k_test_sections_alignment")


def secondlargest(nums):
    largest = -1
    secondlarge = -2
    for index in range(0, len(nums)):
        if nums[index] > largest:
            secondlarge = largest
            largest = nums[index]
        elif nums[index] > secondlarge:
            secondlarge = nums[index]
    return secondlarge


def thirdlargest(nums):
    largest = -1
    secondlarge = -2
    thirdlarge = -3
    for index in range(0, len(nums)):
        if nums[index] > largest:
            thirdlarge = secondlarge
            secondlarge = largest
            largest = nums[index]
        elif nums[index] > secondlarge:
            thirdlarge = secondlarge
            secondlarge = nums[index]
        elif nums[index] > thirdlarge:
            thirdlarge = nums[index]
    return thirdlarge


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


def getcenter(slicenumber, mfovnumber, data):
    xlocsum, ylocsum, nump = 0, 0, 0
    for num in range(1, 62):
        jsonindex = (mfovnumber - 1) * 61 + num - 1
        xlocsum += data[jsonindex]["bbox"][0] + data[jsonindex]["bbox"][1]
        ylocsum += data[jsonindex]["bbox"][2] + data[jsonindex]["bbox"][3]
        nump += 2
    return [xlocsum / nump, ylocsum / nump]


def reorienttris(trilist, pointlist):
    for num in range(0, trilist.shape[0]):
        v0 = np.array(pointlist[trilist[num][0]])
        v1 = np.array(pointlist[trilist[num][1]])
        v2 = np.array(pointlist[trilist[num][2]])
        if np.cross((v1 - v0), (v2 - v0)) < 0:
            trilist[num][0], trilist[num][1] = trilist[num][1], trilist[num][0]
    return


def analyzemfov(slicenumber, mfovnumber, maximgs, data):
    allpoints = np.array([]).reshape((0, 2))
    allresps = []
    alldescs = []
    for i in range(1, maximgs + 1):
        (tempoints, tempresps, tempdescs) = analyzeimg(slicenumber, mfovnumber, i, data)
        allpoints = np.append(allpoints, tempoints, axis=0)
        allresps += tempresps
        alldescs += tempdescs
    allpoints = np.array(allpoints)
    return (allpoints, allresps, alldescs)


def generatematches_cv2(allpoints1, allpoints2, alldescs1, alldescs2, conf):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(np.array(alldescs1), np.array(alldescs2), k=2)
    goodmatches = []
    for m, n in matches:
        if m.distance / n.distance < conf["prelim_matching_args"]["ROD_cutoff"]:
            goodmatches.append([m])
    match_points = np.array([
        np.array([allpoints1[[m[0].queryIdx for m in goodmatches]]][0]),
        np.array([allpoints2[[m[0].trainIdx for m in goodmatches]]][0])])
    return match_points


def generatematches_brute(allpoints1, allpoints2, alldescs1, alldescs2, conf):
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
        if bestdistsofar / secondbestdistsofar < conf["prelim_matching_args"]["ROD_cutoff"]:
            bestpoints1.append(selectedpoint)
            bestpoints2.append(bestcomparedpoint)
    match_points = np.array([bestpoints1, bestpoints2])
    return match_points


def analyze2slicesmfovs(slice1, mfov1, slice2, mfov2, data1, data2, conf):
    print str(slice1) + "-" + str(mfov1) + " vs. " + str(slice2) + "-" + str(mfov2)
    (allpoints1, allresps1, alldescs1) = analyzemfov(slice1, mfov1, 61, data1)
    (allpoints2, allresps2, alldescs2) = analyzemfov(slice2, mfov2, 61, data2)
    match_points = generatematches_cv2(allpoints1, allpoints2, alldescs1, alldescs2, conf)
    model_index = conf["RANSAC_args"]["model_index"]
    iterations = conf["RANSAC_args"]["iterations"]
    max_epsilon = conf["RANSAC_args"]["max_epsilon"]
    min_inlier_ratio = conf["RANSAC_args"]["min_inlier_ratio"]
    min_num_inlier = conf["RANSAC_args"]["min_num_inlier"]
    max_trust = conf["RANSAC_args"]["max_trust"]
    model, filtered_matches = ransac.filter_matches(match_points, model_index, iterations, max_epsilon, min_inlier_ratio, min_num_inlier, max_trust)
    if filtered_matches is None:
        filtered_matches = np.zeros((0, 0))
    return (model, filtered_matches.shape[1], float(filtered_matches.shape[1]) / match_points.shape[1], match_points.shape[1], len(allpoints1), len(allpoints2))


def analyze2slices(slice1, slice2, data1, data2, nummfovs1, nummfovs2, conf):
    toret = []
    modelarr = np.zeros((nummfovs1, nummfovs2), dtype=models.RigidModel)
    numfilterarr = np.zeros((nummfovs1, nummfovs2))
    filterratearr = np.zeros((nummfovs1, nummfovs2))
    besttransform = None

    randomchoices = []
    trytimes = conf["prelim_matching_args"]["trytime"]
    timestorand = trytimes * max(nummfovs1, nummfovs2)
    timesrandtried = 0
    for i in range(0, nummfovs1):
        for j in range(0, nummfovs2):
            randomchoices.append((i + 1, j + 1))

    while (besttransform is None) and (len(randomchoices) > 0):
        randind = random.randint(1, len(randomchoices)) - 1
        mfovcomppicked = randomchoices[randind]
        mfov1, mfov2 = mfovcomppicked
        randomchoices.remove(mfovcomppicked)
        timesrandtried = timesrandtried + 1

        (model, num_filtered, filter_rate, num_rod, num_m1, num_m2) = analyze2slicesmfovs(slice1, mfov1, slice2, mfov2, data1, data2, conf)
        modelarr[mfov1 - 1, mfov2 - 1] = model
        numfilterarr[mfov1 - 1, mfov2 - 1] = num_filtered
        filterratearr[mfov1 - 1, mfov2 - 1] = filter_rate
        if num_filtered > conf["prelim_matching_args"]["numfiltered_cutoff"] and filter_rate > conf["prelim_matching_args"]["filterrate_cutoff"]:
            besttransform = model.get_matrix()
            break
        if timesrandtried > timestorand:
            return toret
    print "Preliminary Transform Found"
    if besttransform is None:
        return toret

    for i in range(0, nummfovs1):
        mycenter = getcenter(slice1, i + 1, data1)
        mycentertrans = np.dot(besttransform, np.append(mycenter, [1]))[0:2]
        distances = np.zeros(nummfovs2)
        for j in range(0, nummfovs2):
            distances[j] = np.linalg.norm(mycentertrans - getcenter(slice2, j + 1, data2))
        checkindices = distances.argsort()[0:7]
        for j in range(0, len(checkindices)):
            (model, num_filtered, filter_rate, num_rod, num_m1, num_m2) = analyze2slicesmfovs(slice1, i + 1, slice2, checkindices[j] + 1, data1, data2, conf)
            modelarr[i, checkindices[j]] = model
            numfilterarr[i, checkindices[j]] = num_filtered
            filterratearr[i, checkindices[j]] = filter_rate
            if num_filtered > conf["prelim_matching_args"]["numfiltered_cutoff"] and filter_rate > conf["prelim_matching_args"]["filterrate_cutoff"]:
                besttransform = model.get_matrix()
                dictentry = {}
                dictentry['mfov1'] = i + 1
                dictentry['mfov2'] = checkindices[j] + 1
                dictentry['features_in_mfov1'] = num_m1
                dictentry['features_in_mfov2'] = num_m2
                dictentry['transformation'] = {
                    "className": model.class_name,
                    "matrix": besttransform.tolist()
                }
                dictentry['matches_rod'] = num_rod
                dictentry['matches_model'] = num_filtered
                dictentry['filter_rate'] = filter_rate
                toret.append(dictentry)
                break
    return toret


def main():
    script, slice1, slice2, datadir, outdir, conffile = sys.argv
    starttime = time.clock()
    slice1 = int(slice1)
    slice2 = int(slice2)
    slicestring1 = ("%03d" % slice1)
    slicestring2 = ("%03d" % slice2)

    os.chdir(datadir)
    with open("tilespecs_after_rotations/W01_Sec" + slicestring1 + ".json") as data_file1:
        data1 = json.load(data_file1)
    with open("tilespecs_after_rotations/W01_Sec" + slicestring2 + ".json") as data_file2:
        data2 = json.load(data_file2)
    with open(conffile) as conf_file:
        conf = json.load(conf_file)
    nummfovs1 = len(data1) / 61
    nummfovs2 = len(data2) / 61

    retval = analyze2slices(slice1, slice2, data1, data2, nummfovs1, nummfovs2, conf)

    jsonfile = {}
    jsonfile['tilespec1'] = "file://" + os.getcwd() + "/tilespecs_after_rotations/W01_Sec" + ("%03d" % slice1) + ".json"
    jsonfile['tilespec2'] = "file://" + os.getcwd() + "/tilespecs_after_rotations/W01_Sec" + ("%03d" % slice2) + ".json"
    jsonfile['matches'] = retval
    jsonfile['runtime'] = time.clock() - starttime
    os.chdir(outdir)
    json.dump(jsonfile, open("Prelim_Slice" + str(slice1) + "vs" + str(slice2) + ".json", 'w'), indent=4)

    '''
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

    %matplotlib
    plt.figure(1)
    for i in range(0,len(pointmatches)):
        point1, point2 = pointmatches[i]
        point1 = np.matrix(point1 - centroid1).dot(R.T).tolist()[0]
        point2 = point2 - centroid2
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]])
        axis('equal')
    '''

if __name__ == '__main__':
    main()
