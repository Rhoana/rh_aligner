# Setup
from __future__ import print_function
import os
import numpy as np
import json
import math
import time
import sys
from scipy.spatial import distance
from scipy import spatial
import cv2
import argparse
import utils
from bounding_box import BoundingBox
import models
import PMCC_filter_example
import multiprocessing as mp
import pyximport
pyximport.install()
import cv_wrap_module


def get_image_top_left(ts, tile_index):
    xloc = ts[tile_index]["bbox"][0]
    yloc = ts[tile_index]["bbox"][2]
    return [xloc, yloc]


def get_mfov_centers_from_json(indexed_ts):
    mfov_centers = {}
    for mfov in indexed_ts.keys():
        tile_bboxes = []
        mfov_tiles = indexed_ts[mfov].values()
        tile_bboxes = zip(*[tile["bbox"] for tile in mfov_tiles])
        min_x = min(tile_bboxes[0])
        max_x = max(tile_bboxes[1])
        min_y = min(tile_bboxes[2])
        max_y = max(tile_bboxes[3])
        # center = [(min_x + min_y) / 2.0, (min_y + max_y) / 2.0], but w/o overflow
        mfov_centers[mfov] = np.array([(min_x / 2.0 + max_x / 2.0, min_y / 2.0 + max_y / 2.0)])
    return mfov_centers


def get_best_transformations(pre_mfov_matches):
    """Returns a dictionary that maps an mfov number to a matrix that best describes the transformation to the other section.
       As not all mfov's may be matched, some mfovs will be missing from the dictionary"""
    transforms = {}
    for m in pre_mfov_matches["matches"]:
        transforms[m["mfov1"]] = m["transformation"]["matrix"]
    return transforms


def find_best_mfov_transformation(mfov, best_transformations, mfov_centers):
    """Returns a matrix that represnets the best transformation for a given mfov to the other section"""
    if mfov in best_transformations.keys():
        return best_transformations[mfov]
    # Need to find a more intelligent way to do this, but this suffices for now
    # Uses transformation of another mfov (should be changed to the closest, unvisited mfov)
    mfov_center = mfov_centers[mfov]
    trans_keys = best_transformations.keys()
    closest_mfov_index = np.argmin([distance.euclidean(mfov_center, mfov_centers[mfovk]) for mfovk in trans_keys])
    # ** Should we add this transformation? maybe not, because we don't want to get a different result when a few
    # ** missing mfov matches occur and the "best transformation" can change when the centers are updated
    return best_transformations[trans_keys[closest_mfov_index]]


def get_tile_centers_from_json(ts):
    tiles_centers = []
    for tile in ts:
        center_x = (tile["bbox"][0] + tile["bbox"][1]) / 2.0
        center_y = (tile["bbox"][2] + tile["bbox"][3]) / 2.0
        tiles_centers.append(np.array([center_x, center_y]))
    return tiles_centers


def get_closest_index_to_point(point, centerstree):
    distanc, closest_index = centerstree.query(point)
    return closest_index


def get_img_matches(ts1, tile_centers1, ts2, tile_centers2, best_transformations, mfov_centers1):
    """For each tile in section1 find the closest 10 tiles in the second image (after applying the preliminary transformation)"""
    img_matches = []

    # TODO - Build a kd-tree of section2 centers

    # Iterate over the first section tiles, and get the approximated location on section 2 (after transformation)
    # and then get the closest 10 tiles (according to their centers) to that location
    for ind1, tile1 in enumerate(ts1):
        center1 = tile_centers1[ind1]
        trans_matrix = find_best_mfov_transformation(tile1["mfov"], best_transformations, mfov_centers1)
        expected_new_center = np.array(np.dot(trans_matrix, np.append(center1, [1]))[0:2])
        distances_to_sec2_mfovs = [np.linalg.norm(expected_new_center - tile_center) for tile_center in tile_centers2]
        closest_indices = np.array(distances_to_sec2_mfovs).argsort()[0:10]
        img_matches.append(closest_indices)
    return img_matches


def is_point_in_img(tile_ts, point):
    """Returns True if the given point lies inside the image as denoted by the given tile_tilespec"""
    # TODO - instead of checking inside the bbox, need to check inside the polygon after transformation
    img_bbox = tile_ts["bbox"]

    if point[0] > img_bbox[0] and point[1] > img_bbox[2] and \
       point[0] < img_bbox[1] and point[1] < img_bbox[3]:
        return True
    return False


def get_images_from_indices_and_point(ts, img_indices, point):
    """Returns all the images at the given img_indices that are overlapping with the given point"""
    img_arr = []
    for img_ind in img_indices:
        if is_point_in_img(ts[img_ind], point):
            img_url = ts[img_ind]["mipmapLevels"]["0"]["imageUrl"]
            img_url = img_url.replace("file://", "")
            img = cv2.imread(img_url, 0)
            img_arr.append((img, img_ind))
    return img_arr


def get_images_from_indices(ts, img_indices):
    """Returns all the images at the given img_indices that are overlapping with the given point"""
    img_arr = []
    for img_ind in img_indices:
        img_url = ts[img_ind]["mipmapLevels"]["0"]["imageUrl"]
        img_url = img_url.replace("file://", "")
        img = cv2.imread(img_url, 0)
        img_arr.append((img, img_ind))
    return img_arr


def get_images_indices_from_indices_and_point(ts, img_indices, point):
    """Returns all the images at the given img_indices that are overlapping with the given point"""
    img_arr = []
    for img_ind in img_indices:
        if is_point_in_img(ts[img_ind], point):
            img_arr.append(img_ind)
    return img_arr


def get_template_from_img_and_point(img1resized, template_size, centerpoint):
    imgheight = img1resized.shape[1]
    imgwidth = img1resized.shape[0]
    notonmesh = False

    xstart = centerpoint[0] - template_size / 2
    ystart = centerpoint[1] - template_size / 2
    xend = centerpoint[0] + template_size / 2
    yend = centerpoint[1] + template_size / 2

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
    return (img1resized[xstart:(xstart + template_size), ystart:(ystart + template_size)].copy(), xstart, ystart, notonmesh)


def get_blank_template_from_img_and_point(imgwidth, imgheight, template_size, centerpoint):
    notonmesh = False

    xstart = centerpoint[0] - template_size / 2
    ystart = centerpoint[1] - template_size / 2
    xend = centerpoint[0] + template_size / 2
    yend = centerpoint[1] + template_size / 2

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
    return (xstart, ystart, template_size, template_size, notonmesh)


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


def perform_pmcc(ts1, ts2, template_size, scaling, img1_ind, best_transformations, mfov_centers1, prelimdict, min_corr, max_curvature, max_rod):
    img1_url = ts1[img1_ind]["mipmapLevels"]["0"]["imageUrl"]
    img1_url = img1_url.replace("file://", "")
    img1 = cv2.imread(img1_url, 0)
    img1_resized = cv2.resize(img1, (0, 0), fx=scaling, fy=scaling)
    img1_offset = get_image_top_left(ts1, img1_ind)
    expected_transform = find_best_mfov_transformation(ts1[img1_ind]["mfov"], best_transformations, mfov_centers1)

    # Loop over all expected img2 indices
    img2_inds = prelimdict[img1_ind].keys()
    img2s = get_images_from_indices(ts2, img2_inds)

    for (img2, img2_ind) in img2s:
        # Resize and find information on the second image
        img2_resized = cv2.resize(img2, (0, 0), fx=scaling/1, fy=scaling/1)
        # img2_offset = get_image_top_left(ts2, img2_ind) - UNUSED
        point_matches = []

        for i in prelimdict[img1_ind][img2_ind]:
            # Get the template for the first Image
            img1_template = get_template_from_img_and_point(img1_resized, template_size, (np.array(i) - img1_offset) * scaling)
            if (img1_template is None):
                continue

            # Find the template coordinates and rotate the template to match preliminary transformation
            chosen_template, startx, starty, not_on_mesh = img1_template
            w, h = chosen_template.shape
            # center_point1 = np.array([startx + w / 2, starty + h / 2]) / scaling + img1_offset - UNUSED
            # expected_new_center = np.dot(expected_transform, np.append(center_point1, [1]))[0:2] - UNUSED
            ro, col = chosen_template.shape
            rad2deg = -180 / math.pi
            # TODO - assumes only rigid transformation, should be more general
            angle_of_rot = rad2deg * math.atan2(expected_transform[1][0], expected_transform[0][0])
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle_of_rot, 1)
            rotated_temp1 = cv2.warpAffine(chosen_template, rotation_matrix, (col, ro))
            xaa = int(w / 2.9)
            rotated_and_cropped_temp1 = rotated_temp1[(w / 2 - xaa):(w / 2 + xaa), (h / 2 - xaa):(h / 2 + xaa)]
            neww, newh = rotated_and_cropped_temp1.shape
            # TODO - assumes a single transformation, but there might be more
            img1_model = models.Transforms.from_tilespec(ts1[img1_ind]["transforms"][0])
            img1_center_point = img1_model.apply(np.array([starty + h / 2, startx + w / 2]) / scaling)  # + imgoffset1

            # Do template matching
            result, reason = PMCC_filter_example.PMCC_match(img2_resized, rotated_and_cropped_temp1, min_correlation=min_corr, maximal_curvature_ratio=max_curvature, maximal_ROD=max_rod)
            if result is not None:
                reasonx, reasony = reason
                # TODO - assumes a single transformation, but there might be more
                img2_model = models.Transforms.from_tilespec(ts2[img2_ind]["transforms"][0])
                img2_center_point = img2_model.apply(np.array([reasony + newh / 2, reasonx + neww / 2]) / scaling)  # + imgoffset2
                point_matches.append((img1_center_point, img2_center_point, not_on_mesh))
    return point_matches


def perform_pmcc_unwrapper(arg):
    ts1, ts2, template_size, scaling, img1_ind, best_transformations, mfov_centers1, prelimdict, min_corr, max_curvature, max_rod = arg
    return perform_pmcc(ts1, ts2, template_size, scaling, img1_ind, best_transformations, mfov_centers1, prelimdict, min_corr, max_curvature, max_rod)


def match_layers_pmcc_matching(tiles_fname1, tiles_fname2, pre_matches_fname, out_fname, conf_fname=None, processes_num=1):
    starttime = time.clock()
    print("Loading tilespecs, parameters, and other preliminary information")
    print("Matching {} with {}".format(tiles_fname1, tiles_fname2))

    # Load parameters file
    params = utils.conf_from_file(conf_fname, 'MatchLayersBlockMatching')
    if params is None:
        params = {}
    cv_wrap_module.setNumThreads(1)

    # Parameters for the matching
    hex_spacing = params.get("hexspacing", 1500)
    scaling = params.get("scaling", 0.2)
    template_size = params.get("template_size", 200)
    template_size *= scaling

    # Parameters for PMCC filtering
    min_corr = params.get("min_correlation", 0.2)
    max_curvature = params.get("maximal_curvature_ratio", 10)
    max_rod = params.get("maximal_ROD", 0.9)

    # Read the tilespecs
    ts1 = utils.load_tilespecs(tiles_fname1)
    ts2 = utils.load_tilespecs(tiles_fname2)
    indexed_ts1 = utils.index_tilespec(ts1)

    # Get the tiles centers for each section
    tile_centers1 = get_tile_centers_from_json(ts1)
    tile_centers1tree = spatial.KDTree(tile_centers1)
    tile_centers2 = get_tile_centers_from_json(ts2)
    mfov_centers1 = get_mfov_centers_from_json(indexed_ts1)

    # Load the preliminary matches
    with open(pre_matches_fname, 'r') as data_matches:
        mfov_pre_matches = json.load(data_matches)
    if len(mfov_pre_matches["matches"]) == 0:
        print("No matches were found in pre-matching")
        return
    best_transformations = get_best_transformations(mfov_pre_matches)

    # Create output dictionary
    out_jsonfile = {}
    out_jsonfile['tilespec1'] = tiles_fname1
    out_jsonfile['tilespec2'] = tiles_fname2

    # Get width and height of a resized template
    img_matches = get_img_matches(ts1, tile_centers1, ts2, tile_centers2, best_transformations, mfov_centers1)
    img1_url = ts1[50]["mipmapLevels"]["0"]["imageUrl"]
    img1_url = img1_url.replace("file://", "")
    img1 = cv2.imread(img1_url, 0)
    img1_resized = cv2.resize(img1, (0, 0), fx=scaling, fy=scaling)
    img1width = img1_resized.shape[0]
    img1height = img1_resized.shape[1]

    # Generate an hexagonal grid according to the first section's bounding box
    print("Generating Hexagonal Grid")
    bb = BoundingBox.read_bbox(tiles_fname1)
    hexgr = generatehexagonalgrid(bb, hex_spacing)

    # Iterate over the hexagonal points to find preliminary matches
    print("Starting finding preliminary matches for each point")
    prelimdict = {}
    for i in range(len(hexgr)):
        # Find the tile image where the point from the hexagonal is in the first section
        img1_ind = get_closest_index_to_point(hexgr[i], tile_centers1tree)
        if img1_ind is None:
            continue
        if not is_point_in_img(ts1[img1_ind], hexgr[i]):
            continue
        # Get expected point of hexgr[i] in the second section
        img1_offset = get_image_top_left(ts1, img1_ind)
        expected_transform = find_best_mfov_transformation(ts1[img1_ind]["mfov"], best_transformations, mfov_centers1)
        img1_template = get_blank_template_from_img_and_point(img1width, img1height, template_size, (np.array(hexgr[i]) - img1_offset) * scaling)
        if img1_template is None:
            continue
        startx, starty, w, h, not_on_mesh = img1_template
        center_point1 = np.array([startx + w / 2, starty + h / 2]) / scaling + img1_offset
        expected_new_center = np.dot(expected_transform, np.append(center_point1, [1]))[0:2]
        img2_inds = img_matches[img1_ind]
        img2s = get_images_indices_from_indices_and_point(ts2, img2_inds, expected_new_center)

        for img2_ind in img2s:
            # Build dictionary here
            if img1_ind in prelimdict:
                newdi = prelimdict[img1_ind]
                if img2_ind in newdi:
                    newdi[img2_ind].append(hexgr[i])
                else:
                    newdi[img2_ind] = []
                    newdi[img2_ind].append(hexgr[i])
            else:
                newdi = {}
                newdi[img2_ind] = []
                newdi[img2_ind].append(hexgr[i])
                prelimdict[img1_ind] = newdi

    # Generate arguments for PMCC matchingf
    print("Generating PMCC Matching Arguments")
    commandlist = []
    point_matches = []
    for img1_ind in prelimdict.keys():
        commandlist.append((ts1, ts2, template_size, scaling, img1_ind, best_transformations, mfov_centers1, prelimdict, min_corr, max_curvature, max_rod))

    # Execute PMCC Matching
    print("Performing PMCC Matching")
    pool = mp.Pool(processes=processes_num)
    results = pool.map(perform_pmcc_unwrapper, commandlist)
    point_matches = [item for sublist in results for item in sublist]

    # Save the output
    print("Saving output file")
    out_jsonfile['runtime'] = time.clock() - starttime
    out_jsonfile['mesh'] = hexgr

    final_point_matches = []
    for pm in point_matches:
        p1, p2, nmesh = pm
        record = {}
        record['point1'] = p1.tolist()
        record['point2'] = p2.tolist()
        record['isvirtualpoint'] = nmesh
        final_point_matches.append(record)

    out_jsonfile['pointmatches'] = final_point_matches
    with open(out_fname, 'w') as out:
        json.dump(out_jsonfile, out, indent=4)

    print("Done")


def main():
    # print(sys.argv)
    # Command line parser
    parser = argparse.ArgumentParser(description='Given two tilespecs of two sections, and a preliminary matches list, generates a grid the image, and performs block matching (with PMCC filtering).')
    parser.add_argument('tiles_file1', metavar='tiles_file1', type=str,
                        help='the first layer json file containing tilespecs')
    parser.add_argument('tiles_file2', metavar='tiles_file2', type=str,
                        help='the second layer json file containing tilespecs')
    parser.add_argument('pre_matches_file', metavar='pre_matches_file', type=str,
                        help='a json file that contains the preliminary matches')
    parser.add_argument('-o', '--output_file', type=str,
                        help='an output correspondent_spec file, that will include the matches between the sections (default: ./matches.json)',
                        default='./matches.json')
    parser.add_argument('-c', '--conf_file_name', type=str,
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)
    parser.add_argument('-t', '--threads_num', type=int,
                        help='the number of threads (processes) to use (default: 1)',
                        default=1)

    args = parser.parse_args()
    match_layers_pmcc_matching(args.tiles_file1, args.tiles_file2,
                               args.pre_matches_file, args.output_file,
                               conf_fname=args.conf_file_name, processes_num=args.threads_num)

if __name__ == '__main__':
    main()
