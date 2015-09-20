# Setup
from __future__ import print_function
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
import argparse
import utils
# os.chdir("C:/Users/Raahil/Documents/Research2015_eclipse/Testing")
# os.chdir("/data/SCS_2015-4-27_C1w7_alignment")
# os.chdir("/data/jpeg2k_test_sections_alignment")

TILES_PER_MFOV = 61


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


def getnumsfromindex(ind):
    return (ind / TILES_PER_MFOV + 1, ind % TILES_PER_MFOV + 1)


def getindexfromnums((mfovnum, imgnum)):
    return (mfovnum - 1) * TILES_PER_MFOV + imgnum - 1


def load_features(feature_file, tile_ts):
    # Should have the same name as the following: [tilespec base filename]_[img filename].json/.hdf5
    assert(os.path.basename(os.path.splitext(tile_ts["mipmapLevels"]["0"]["imageUrl"])[0]) in feature_file)

    # print("Loading feature file {} of tile {}, with a transform {}".format(feature_file, tile_ts["mipmapLevels"]["0"]["imageUrl"], tile_ts["transforms"][0]))
    # load the image features
    with h5py.File(feature_file, 'r') as f:
        resps = f['pts']['responses'][:]
        descs = f['descs'][:]
        octas = f['pts']['octaves'][:]
        allps = np.array(f['pts']['locations'])

    # If no relevant features are found, return an empty set
    if (len(allps) == 0):
        return (np.array([]).reshape((0, 2)), [], [])

    # Apply the transformation to each point
    newmodel = models.Transforms.from_tilespec(tile_ts["transforms"][0])
    newallps = newmodel.apply_special(allps)

    currentocta = (octas.astype(int) & 0xff)
    currentocta[currentocta > 127] -= 255
    mask = (currentocta == 4) | (currentocta == 5)
    points = newallps[mask, :]
    resps = resps[mask]
    descs = descs[mask]
    return (points, resps, descs)


def getcenter(mfov_ts):
    xlocsum, ylocsum, nump = 0, 0, 0
    for tile_ts in mfov_ts.values():
        xlocsum += tile_ts["bbox"][0] + tile_ts["bbox"][1]
        ylocsum += tile_ts["bbox"][2] + tile_ts["bbox"][3]
        nump += 2
    return [xlocsum / nump, ylocsum / nump]


def getslicecenter(tilespecs):
    xlocsum, ylocsum, nump = 0, 0, 0
    for i in tilespecs:
        mfov_ts = tilespecs[i]
        for tile_ts in mfov_ts.values():
            xlocsum += tile_ts["bbox"][0] + tile_ts["bbox"][1]
            ylocsum += tile_ts["bbox"][2] + tile_ts["bbox"][3]
            nump += 2
    return [xlocsum / nump, ylocsum / nump]


def get_closest_index_to_point(point, centers):
    closest_index = np.argmin([distance.euclidean(point, center) for center in centers])
    return closest_index


def getcenterprecompute(num, precomplist):
    return precomplist[num - 1]


def getdistprecompute(num, precomplist):
    return precomplist[num - 1]


def pickrandomnearcenter(choicelist, tilespecs1, tilespecs2, precompcenters1, precompdists1, avgx, avgy, avgx2, avgy2):
    choices1 = [i[0] for i in choicelist]
    centers1 = [getcenterprecompute(i, precompcenters1) for i in choices1]
    # (avgx, avgy) = getslicecenter(tilespecs1)
    dists1 = [getdistprecompute(i, precompdists1) for i in choices1]
    dists1sum = np.sum(dists1)
    dists1 = [i / dists1sum for i in dists1]
    choice1 = np.random.choice(choices1, p=dists1)

    possiblechoicesprelim = []
    for i in choicelist:
        if i[0] == choice1:
            possiblechoicesprelim.append(i)

    choices2 = [i[1] for i in possiblechoicesprelim]
    centers2 = [getcenter(tilespecs2[i]) for i in choices2]
    # (avgx2, avgy2) = getslicecenter(tilespecs2)
    dists2 = [1 / distance.euclidean((avgx2, avgy2), i) for i in centers2]
    dists2sum = np.sum(dists2)
    dists2 = [i / dists2sum for i in dists2]
    choice2 = np.random.choice(choices2, p=dists2)
    finalchoice = (choice1, choice2)
    return [i for i in range(len(choicelist)) if choicelist[i] == finalchoice][0]


def reorienttris(trilist, pointlist):
    for num in range(0, trilist.shape[0]):
        v0 = np.array(pointlist[trilist[num][0]])
        v1 = np.array(pointlist[trilist[num][1]])
        v2 = np.array(pointlist[trilist[num][2]])
        if np.cross((v1 - v0), (v2 - v0)) < 0:
            trilist[num][0], trilist[num][1] = trilist[num][1], trilist[num][0]
    return


def analyzemfov(mfov_ts, features_dir):
    """Returns all the relevant features of the tiles in a single mfov"""
    allpoints = np.array([]).reshape((0, 2))
    allresps = []
    alldescs = []

    mfov_num = int(mfov_ts.values()[0]["mfov"])
    mfov_string = ("%06d" % mfov_num)
    mfov_feature_files = sorted(glob.glob(os.path.join(os.path.join(features_dir, mfov_string), '*')))
    if len(mfov_feature_files) < TILES_PER_MFOV:
        print("Warning: number of feature files in directory: {} is smaller than {}".format(os.path.join(os.path.join(features_dir, mfov_string)), TILES_PER_MFOV), file=sys.stderr)

    # load each features file, and concatenate all to single lists
    for feature_file in mfov_feature_files:
        # Get the correct tile tilespec from the section tilespec (convert to int to remove leading zeros)
        tile_num = int(feature_file.split('sifts_')[1].split('_')[2])
        (tempoints, tempresps, tempdescs) = load_features(feature_file, mfov_ts[tile_num])
        if type(tempdescs) is not list:
            # concatentate the results
            allpoints = np.append(allpoints, tempoints, axis=0)
            allresps.append(tempresps)
            alldescs.append(tempdescs)
    allpoints = np.array(allpoints)
    return (allpoints, np.concatenate(allresps), np.vstack(alldescs))


def generatematches_cv2(allpoints1, allpoints2, alldescs1, alldescs2, actual_params):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(np.array(alldescs1), np.array(alldescs2), k=2)
    goodmatches = []
    for m, n in matches:
        if m.distance / n.distance < actual_params["ROD_cutoff"]:
            goodmatches.append([m])
    match_points = np.array([
        np.array([allpoints1[[m[0].queryIdx for m in goodmatches]]][0]),
        np.array([allpoints2[[m[0].trainIdx for m in goodmatches]]][0])])
    return match_points


def generatematches_brute(allpoints1, allpoints2, alldescs1, alldescs2, actual_params):
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
        if bestdistsofar / secondbestdistsofar < actual_params["ROD_cutoff"]:
            bestpoints1.append(selectedpoint)
            bestpoints2.append(bestcomparedpoint)
    match_points = np.array([bestpoints1, bestpoints2])
    return match_points


def analyze2slicesmfovs(mfov1_ts, mfov2_ts, features_dir1, features_dir2, actual_params):
    first_tile1 = mfov1_ts.values()[0]
    first_tile2 = mfov2_ts.values()[0]
    print("Sec{}_Mfov{} vs. Sec{}_Mfov{}".format(first_tile1["layer"], first_tile1["mfov"], first_tile2["layer"], first_tile2["mfov"]))
    (allpoints1, allresps1, alldescs1) = analyzemfov(mfov1_ts, features_dir1)
    (allpoints2, allresps2, alldescs2) = analyzemfov(mfov2_ts, features_dir2)
    match_points = generatematches_cv2(allpoints1, allpoints2, alldescs1, alldescs2, actual_params)
    model_index = actual_params["model_index"]
    iterations = actual_params["iterations"]
    max_epsilon = actual_params["max_epsilon"]
    min_inlier_ratio = actual_params["min_inlier_ratio"]
    min_num_inlier = actual_params["min_num_inlier"]
    max_trust = actual_params["max_trust"]
    model, filtered_matches = ransac.filter_matches(match_points, model_index, iterations, max_epsilon, min_inlier_ratio, min_num_inlier, max_trust)
    if filtered_matches is None:
        filtered_matches = np.zeros((0, 0))
    if model is None:
        print("Could not find a valid model between Sec{}_Mfov{} vs. Sec{}_Mfov{}".format(first_tile1["layer"], first_tile1["mfov"], first_tile2["layer"], first_tile2["mfov"]))
    else:
        print("Found a model {} (with {} matches) between Sec{}_Mfov{} vs. Sec{}_Mfov{}".format(model.to_str(), filtered_matches.shape[1], first_tile1["layer"], first_tile1["mfov"], first_tile2["layer"], first_tile2["mfov"]))
    return (model, filtered_matches.shape[1], float(filtered_matches.shape[1]) / match_points.shape[1], match_points.shape[1], len(allpoints1), len(allpoints2))


def analyze2slices(indexed_ts1, indexed_ts2, nummfovs1, nummfovs2, features_dir1, features_dir2, actual_params):
    layer1 = indexed_ts1.values()[0].values()[0]["layer"]
    layer2 = indexed_ts2.values()[0].values()[0]["layer"]
    toret = []
    modelarr = np.zeros((nummfovs1, nummfovs2), dtype=models.RigidModel)
    numfilterarr = np.zeros((nummfovs1, nummfovs2))
    filterratearr = np.zeros((nummfovs1, nummfovs2))
    besttransform = None

    randomchoices = []
    trytimes = actual_params["max_attempts"]
    timestorand = trytimes * max(nummfovs1, nummfovs2)
    timesrandtried = 0
    for i in range(0, nummfovs1):
        for j in range(0, nummfovs2):
            randomchoices.append((i + 1, j + 1))
    precompcenters1 = [getcenter(indexed_ts1[i]) for i in range(1, nummfovs1 + 1)]
    (avgx, avgy) = getslicecenter(indexed_ts1)
    (avgx2, avgy2) = getslicecenter(indexed_ts2)
    precompdists1 = [1 / distance.euclidean((avgx, avgy), i) for i in precompcenters1]

    while (besttransform is None) and (len(randomchoices) > 0):
        # randind = random.randint(1, len(randomchoices)) - 1
        randind = pickrandomnearcenter(randomchoices, indexed_ts1, indexed_ts2, precompcenters1, precompdists1, avgx, avgy, avgx2, avgy2)
        mfovcomppicked = randomchoices[randind]
        mfov1, mfov2 = mfovcomppicked
        randomchoices.remove(mfovcomppicked)
        timesrandtried = timesrandtried + 1

        (model, num_filtered, filter_rate, num_rod, num_m1, num_m2) = analyze2slicesmfovs(indexed_ts1[mfov1], indexed_ts2[mfov2], features_dir1, features_dir2, actual_params)
        modelarr[mfov1 - 1, mfov2 - 1] = model
        numfilterarr[mfov1 - 1, mfov2 - 1] = num_filtered
        filterratearr[mfov1 - 1, mfov2 - 1] = filter_rate
        if num_filtered > actual_params["num_filtered_cutoff"] and filter_rate > actual_params["filter_rate_cutoff"]:
            besttransform = model.get_matrix()
            break
        if timesrandtried > timestorand:
            print("Max attempts to find a preliminary transform reached, no model found between sections: {} and {}".format(layer1, layer2))
            return toret

    if besttransform is None:
        print("Could not find a preliminary transform between sections: {} and {}".format(layer1, layer2))
        return toret

    print("Found a preliminary transform between sections: {} and {}, with model: {}".format(layer1, layer2, besttransform))

    for i in range(0, nummfovs1):
        mycenter = getcenter(indexed_ts1[i + 1])
        mycentertrans = np.dot(besttransform, np.append(mycenter, [1]))[0:2]
        distances = np.zeros(nummfovs2)
        for j in range(0, nummfovs2):
            distances[j] = np.linalg.norm(mycentertrans - getcenter(indexed_ts2[j + 1]))
        checkindices = distances.argsort()[0:7]
        for j in range(0, len(checkindices)):
            (model, num_filtered, filter_rate, num_rod, num_m1, num_m2) = analyze2slicesmfovs(indexed_ts1[i + 1], indexed_ts2[checkindices[j] + 1], features_dir1, features_dir2, actual_params)
            modelarr[i, checkindices[j]] = model
            numfilterarr[i, checkindices[j]] = num_filtered
            filterratearr[i, checkindices[j]] = filter_rate
            if num_filtered > actual_params["num_filtered_cutoff"] and filter_rate > actual_params["filter_rate_cutoff"]:
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


def match_layers_sift_features(tiles_fname1, features_dir1, tiles_fname2, features_dir2, out_fname, conf_fname=None):
    params = utils.conf_from_file(conf_fname, 'MatchLayersSiftFeaturesAndFilter')
    if params is None:
        params = {}
    actual_params = {}
    # Parameters for the matching
    actual_params["max_attempts"] = params.get("max_attempts", 10)
    actual_params["num_filtered_cutoff"] = params.get("num_filtered_cutoff", 50)
    actual_params["filter_rate_cutoff"] = params.get("filter_rate_cutoff", 0.25)
    actual_params["ROD_cutoff"] = params.get("ROD_cutoff", 0.92)

    # Parameters for the RANSAC
    actual_params["model_index"] = params.get("model_index", 1)
    actual_params["iterations"] = params.get("iterations", 500)
    actual_params["max_epsilon"] = params.get("max_epsilon", 500.0)
    actual_params["min_inlier_ratio"] = params.get("min_inlier_ratio", 0.01)
    actual_params["min_num_inlier"] = params.get("min_num_inliers", 7)
    actual_params["max_trust"] = params.get("max_trust", 3)

    print("Matching layers: {} and {}".format(tiles_fname1, tiles_fname2))

    starttime = time.clock()

    # Read the tilespecs
    indexed_ts1 = utils.index_tilespec(utils.load_tilespecs(tiles_fname1))
    indexed_ts2 = utils.index_tilespec(utils.load_tilespecs(tiles_fname2))

    num_mfovs1 = len(indexed_ts1)
    num_mfovs2 = len(indexed_ts2)

    # Match the two sections
    retval = analyze2slices(indexed_ts1, indexed_ts2, num_mfovs1, num_mfovs2, features_dir1, features_dir2, actual_params)

    # Save the output
    jsonfile = {}
    jsonfile['tilespec1'] = tiles_fname1
    jsonfile['tilespec2'] = tiles_fname2
    jsonfile['matches'] = retval
    jsonfile['runtime'] = time.clock() - starttime
    with open(out_fname, 'w') as out:
        json.dump(jsonfile, out, indent=4)
    print("Done.")


def main():
    print(sys.argv)
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over the mfovs in 2 tilespecs of two sections, computing matches for each overlapping mfov.')
    parser.add_argument('tiles_file1', metavar='tiles_file1', type=str,
                        help='the first layer json file containing tilespecs')
    parser.add_argument('features_dir1', metavar='features_dir1', type=str,
                        help='the first layer features directory')
    parser.add_argument('tiles_file2', metavar='tiles_file2', type=str,
                        help='the second layer json file containing tilespecs')
    parser.add_argument('features_dir2', metavar='features_dir2', type=str,
                        help='the second layer features directory')
    parser.add_argument('-o', '--output_file', type=str,
                        help='an output correspondent_spec file, that will include the matches between the sections (default: ./matches.json)',
                        default='./matches.json')
    parser.add_argument('-c', '--conf_file_name', type=str,
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)

    args = parser.parse_args()

    match_layers_sift_features(args.tiles_file1, args.features_dir1,
                               args.tiles_file2, args.features_dir2, args.output_file,
                               conf_fname=args.conf_file_name)


if __name__ == '__main__':
    main()
