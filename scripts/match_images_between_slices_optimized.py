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
from scipy import spatial


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

def get_best_transformations(pre_mfov_matches, tiles_fname1, tiles_fname2, mfov_centers1, mfov_centers2):
    """Returns a dictionary that maps an mfov number to a matrix that best describes the transformation to the other section.
       As not all mfov's may be matched, some mfovs will be missing from the dictionary.
       If the given tiles file names are reversed, an inverted matrix is returned."""
    transforms = {}
    reversed = False
    if tiles_fname1 == pre_mfov_matches["tilespec1"] and \
       tiles_fname2 == pre_mfov_matches["tilespec2"]:
        reversed = False
    elif tiles_fname1 == pre_mfov_matches["tilespec2"] and \
         tiles_fname2 == pre_mfov_matches["tilespec1"]:
        reversed = True
    else:
        print("Error: could not find pre_matches between tilespecs {} and {} (found tilespecs {} and {} instead!).".format(tiles_fname1, tiles_fname2, pre_mfov_matches["tilespec1"], pre_mfov_matches["tilespec2"]), file=sys.stderr)
        return {}

    if reversed:
        # for each transformed mfov center from section 1 match the reversed transformation matrix (from section 2 to section 1)
        transformed_section_centers2 = [np.dot(m["transformation"]["matrix"], np.append(mfov_centers2[m["mfov1"]], 1.0))[:2] for m in pre_mfov_matches["matches"]]
        reversed_transformations = [np.linalg.inv(m["transformation"]["matrix"]) for m in pre_mfov_matches["matches"]]

        # Build a kdtree from the mfovs centers in section 2
        kdtree = spatial.KDTree(np.array(mfov_centers1.values()).flatten().reshape(len(mfov_centers1), 2))

        # For each mfov transformed center in section 2, find the closest center, and declare it as a transformation
        closest_centers_idx = kdtree.query(transformed_section_centers2)[1]
        
        assert(len(reversed_transformations) == len(closest_centers_idx))
        for closest_idx, reversed_transform in zip(closest_centers_idx, reversed_transformations):
            mfov_num2 = closest_idx + 1
            transforms[mfov_num2] = reversed_transform

    else:
        for m in pre_mfov_matches["matches"]:
            transforms[m["mfov1"]] = m["transformation"]["matrix"]

#    for m in pre_mfov_matches["matches"]:
#        if reversed:
#            cur_matrix = m["transformation"]["matrix"]
#            section_center1 = mfov_centers1[m["mfov1"] - 1]
#            transformed_section_center2 = np.dot(cur_matrix, section_center1)
#            # Reverse the matrix
#            rev_matrix = np.linalg.inv(cur_matrix)
#
#            # Find the mfov that is closest to the one that is being transformed
#            closest_mfov_num2 = np.argmin([((c[0] - transformed_section_center2[0])**2 + (c[1] - transformed_section_center2[1])**2) for c in mfov_centers2]) + 1
#
#            transforms[m["mfov2"]] = rev_matrix
#        else:
#            transforms[m["mfov1"]] = m["transformation"]["matrix"]
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

def get_closest_indices_to_point(point, centerstree, num):
    _, closest_indices = centerstree.query(point, k=num)
    return closest_indices

def get_img_matches(ts1, tile_centers1, ts2, tile_centers2_kdtree, best_transformations, mfov_centers1):
    """For each tile in section1 find the closest 10 tiles in the second image (after applying the preliminary transformation)"""
    img_matches = []

    # Iterate over the first section tiles, and get the approximated location on section 2 (after transformation)
    # and then get the closest 10 tiles (according to their centers) to that location
    for ind1, tile1 in enumerate(ts1):
        center1 = tile_centers1[ind1]
        trans_matrix = find_best_mfov_transformation(tile1["mfov"], best_transformations, mfov_centers1)
        expected_new_center = np.array(np.dot(trans_matrix, np.append(center1, [1]))[0:2])
        closest_indices = get_closest_indices_to_point(expected_new_center, tile_centers2_kdtree, 10)
        img_matches.append(closest_indices)
    return img_matches

def get_mfov_img_matches(ts1, tile_centers1, ts2, tile_centers2_kdtree, best_transformations, mfov_centers1, targeted_mfov1):
    """For each tile in the targeted_mfov1 of section1 find the closest 10 tiles in the second image (after applying the preliminary transformation)"""
    img_matches = []

    # Iterate over the first section tiles, and get the approximated location on section 2 (after transformation)
    # and then get the closest 10 tiles (according to their centers) to that location
    for ind1, tile1 in enumerate(ts1):
        if tile1["mfov"] != targeted_mfov1:
            img_matches.append([])
            continue
        center1 = tile_centers1[ind1]
        trans_matrix = find_best_mfov_transformation(tile1["mfov"], best_transformations, mfov_centers1)
        expected_new_center = np.array(np.dot(trans_matrix, np.append(center1, [1]))[0:2])
        closest_indices = get_closest_indices_to_point(expected_new_center, tile_centers2_kdtree, 10)
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


def get_images_from_indices(ts, img_indices):
    """Returns all the images at the given img_indices that are overlapping with the given point"""
    img_arr = []
    for img_ind in img_indices:
        img_url = ts[img_ind]["mipmapLevels"]["0"]["imageUrl"]
        img_url = img_url.replace("file://", "")
        img = cv2.imread(img_url, 0)
        if img is None or img.shape == (0,0):
            print("Error: could not read image {} - Aborting!".format(img_url), file=sys.stderr)
            sys.exit(1)
        #img = cv2.equalizeHist(img)
        img_arr.append((img, img_ind))
    return img_arr


def get_images_indices_from_indices_and_point(ts, img_indices, point):
    """Returns all the images at the given img_indices that are overlapping with the given point"""
    img_arr = [img_ind for img_ind in img_indices if is_point_in_img(ts[img_ind], point)]
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
            pointsret.append([int(xpos + boundingbox[0]), int(ypos + boundingbox[2])])
    return pointsret


def perform_pmcc(ts1, ts2, template_size, scaling, img1_ind, best_transformations, mfov_centers1, prelimdict, min_corr, max_curvature, max_rod, debug_save_matches=False, debug_dir=None):

    point_matches = []
    img1_url = ts1[img1_ind]["mipmapLevels"]["0"]["imageUrl"]
    img1_url = img1_url.replace("file://", "")
    img1 = cv2.imread(img1_url, 0)
    if img1 is None or img1.shape == (0,0):
        print("Error: could not read image {} - Aborting!".format(img1_url), file=sys.stderr)
        sys.exit(1)
    #img1 = cv2.equalizeHist(img1)
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

        for hexgr_point in prelimdict[img1_ind][img2_ind]:
            # Get the template for the first Image
            img1_template = get_template_from_img_and_point(img1_resized, template_size, (np.array(hexgr_point) - img1_offset) * scaling)
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
            result, reason, match_val = PMCC_filter_example.PMCC_match(img2_resized, rotated_and_cropped_temp1, min_correlation=min_corr, maximal_curvature_ratio=max_curvature, maximal_ROD=max_rod)
            if result is not None:
                reasonx, reasony = reason
                # TODO - assumes a single transformation, but there might be more
                img2_model = models.Transforms.from_tilespec(ts2[img2_ind]["transforms"][0])
                img2_center_point = img2_model.apply(np.array([reasony + newh / 2, reasonx + neww / 2]) / scaling)  # + imgoffset2
                point_matches.append((img1_center_point, img2_center_point, not_on_mesh, match_val))
                if debug_save_matches:
                    #debug_out_fname1 = os.path.join(debug_dir, "debug_match_sec1{}-{}_sec2{}-{}_image1.png".format(hexgr_point[0], hexgr_point[1], reasonx, reasony))
                    #debug_out_fname2 = os.path.join(debug_dir, "debug_match_sec1{}-{}_sec2{}-{}_image2.png".format(hexgr_point[0], hexgr_point[1], reasonx, reasony))
                    debug_out_fname1 = os.path.join(debug_dir, "debug_match_sec1{}-{}_sec2{}-{}_image1.png".format(int(img1_center_point[0]), int(img1_center_point[1]), int(img2_center_point[0]), int(img2_center_point[1])))
                    debug_out_fname2 = os.path.join(debug_dir, "debug_match_sec1{}-{}_sec2{}-{}_image2.png".format(int(img1_center_point[0]), int(img1_center_point[1]), int(img2_center_point[0]), int(img2_center_point[1])))
                    cv2.imwrite(debug_out_fname1, rotated_and_cropped_temp1)
                    temp1_final_sizex = rotated_and_cropped_temp1.shape[0]
                    temp1_final_sizey = rotated_and_cropped_temp1.shape[1]
                    img2_cut_out = img2_resized[reasonx:(reasonx + temp1_final_sizex), reasony:(reasony + temp1_final_sizey)]
                    cv2.imwrite(debug_out_fname2, img2_cut_out)
                # no need to find another possible point
    return point_matches


def perform_pmcc_unwrapper(arg):
    ts1, ts2, template_size, scaling, img1_ind, best_transformations, mfov_centers1, prelimdict, min_corr, max_curvature, max_rod, debug_save_matches, debug_dir = arg
    sys.stdout = sys.__stdout__
    return perform_pmcc(ts1, ts2, template_size, scaling, img1_ind, best_transformations, mfov_centers1, prelimdict, min_corr, max_curvature, max_rod, debug_save_matches, debug_dir)


def match_layers_pmcc_matching(tiles_fname1, tiles_fname2, pre_matches_fname, out_fname, targeted_mfov=-1, conf_fname=None, processes_num=1):
    starttime = time.time()
    print("Loading tilespecs, parameters, and other preliminary information")
    print("Block-Matching+PMCC layers: {} with {} targeted mfov: {}".format(tiles_fname1, tiles_fname2, 'all' if targeted_mfov==-1 else targeted_mfov))

    # Load parameters file
    params = utils.conf_from_file(conf_fname, 'MatchLayersBlockMatching')
    if params is None:
        params = {}
    cv_wrap_module.setNumThreads(1)

    # Parameters for the matching
    hex_spacing = params.get("hex_spacing", 1500)
    scaling = params.get("scaling", 0.2)
    template_size = params.get("template_size", 200)
    template_size *= scaling
    print("Actual template size (after scaling): {}".format(template_size))

    # Parameters for PMCC filtering
    min_corr = params.get("min_correlation", 0.2)
    max_curvature = params.get("maximal_curvature_ratio", 10)
    max_rod = params.get("maximal_ROD", 0.9)

    debug_save_matches = False
    debug_dir = None
    if "debug_save_matches" in params.keys():
        print("Debug mode - on")
        debug_save_matches = True
        if debug_save_matches:
            # Create a debug directory
            import datetime
            debug_dir = os.path.join(os.path.dirname(out_fname), 'debug_matches_{}'.format(datetime.datetime.now().isoformat()))
            os.mkdir(debug_dir)

    # Read the tilespecs
    ts1 = utils.load_tilespecs(tiles_fname1)
    ts2 = utils.load_tilespecs(tiles_fname2)
    indexed_ts1 = utils.index_tilespec(ts1)
    indexed_ts2 = utils.index_tilespec(ts2)

    # Get the tiles centers for each section
    tile_centers1 = get_tile_centers_from_json(ts1)
    tile_centers1tree = spatial.KDTree(tile_centers1)
    tile_centers2 = get_tile_centers_from_json(ts2)
    tile_centers2tree = spatial.KDTree(tile_centers2)
    mfov_centers1 = get_mfov_centers_from_json(indexed_ts1)
    mfov_centers2 = get_mfov_centers_from_json(indexed_ts2)

    # Load the preliminary matches
    with open(pre_matches_fname, 'r') as data_matches:
        mfov_pre_matches = json.load(data_matches)
    if len(mfov_pre_matches["matches"]) == 0:
        print("No matches were found in pre-matching, aborting Block-Matching proces between layers: {} and {}".format(tiles_fname1, tiles_fname2))
        return
    best_transformations = get_best_transformations(mfov_pre_matches, tiles_fname1, tiles_fname2, mfov_centers1, mfov_centers2)

    # Create output dictionary
    out_jsonfile = {}
    out_jsonfile['tilespec1'] = tiles_fname1
    out_jsonfile['tilespec2'] = tiles_fname2

    # Get width and height of a resized template (using the first image size)
    img1height = ts1[0]["width"] * scaling
    img1width = ts1[0]["height"] * scaling
    #img1width = ts1[0]["width"] * scaling
    #img1height = ts1[0]["height"] * scaling
    #img1_url = ts1[0]["mipmapLevels"]["0"]["imageUrl"]
    #img1_url = img1_url.replace("file://", "")
    #img1 = cv2.imread(img1_url, 0)
    #if img1 is None or img1.shape == (0,0):
    #    print("Error: could not read image {} - Aboritng!".format(img1_url), file=sys.stderr)
    #    sys.exit(1)
    #
    #img1_resized = cv2.resize(img1, (0, 0), fx=scaling, fy=scaling)
    #img1width = img1_resized.shape[0]
    #img1height = img1_resized.shape[1]

    # Generate an hexagonal grid according to the first section's bounding box
    print("Generating Hexagonal Grid")
    bb = BoundingBox.read_bbox(tiles_fname1)
    print("bounding_box: ", bb)
    hexgr = generatehexagonalgrid(bb, hex_spacing)
    #print(hexgr)
    # if a single mfov is targeted, restrict the hexagonal grid to that mfov locations
    if targeted_mfov != -1:
        mfov_tiles = indexed_ts1[targeted_mfov]
        bb_mfov = BoundingBox.read_bbox_from_ts(mfov_tiles.values())
        print("Trimming bounding box grid points to {} (mfov {})".format(bb_mfov.toArray(), targeted_mfov))
        hexgr = [p for p in hexgr if bb_mfov.contains(np.array([p]))]

    # Iterate over the hexagonal points to find preliminary matches
    print("Starting finding preliminary matches for each point ({} possible points)".format(len(hexgr)))
    if targeted_mfov == -1:
        img_matches = get_img_matches(ts1, tile_centers1, ts2, tile_centers2tree, best_transformations, mfov_centers1)
    else:
        img_matches = get_mfov_img_matches(ts1, tile_centers1, ts2, tile_centers2tree, best_transformations, mfov_centers1, targeted_mfov)
    # Store all the possible matching tiles between the sections
    # (1 tile from section1 to a list of tiles in section2): tile1 (of section1) -> [tile2_a, tile2_b, ...] (of section2)
    prelimdict = {}
    actual_matches_num = 0
    for i in range(len(hexgr)):
        # Find the tile image where the point from the hexagonal is in the first section
        img1_ind = get_closest_index_to_point(hexgr[i], tile_centers1tree)
        #print(img1_ind)
        if img1_ind is None:
            continue
        if targeted_mfov != -1 and ts1[img1_ind]["mfov"] != targeted_mfov:
            continue
        if not is_point_in_img(ts1[img1_ind], hexgr[i]):
            continue
        # Get expected point of hexgr[i] in the second section
        #img1_offset = get_image_top_left(ts1, img1_ind)
        expected_transform = find_best_mfov_transformation(ts1[img1_ind]["mfov"], best_transformations, mfov_centers1)
        #img1_template = get_blank_template_from_img_and_point(img1width, img1height, template_size, (np.array(hexgr[i]) - img1_offset) * scaling)
        #if img1_template is None:
        #    continue
        #startx, starty, w, h, not_on_mesh = img1_template
        #center_point1 = np.array([startx + w / 2, starty + h / 2]) / scaling + img1_offset
        center_point1 = np.array(hexgr[i])
        expected_new_center = np.dot(expected_transform, np.append(center_point1, [1]))[0:2]
        img2_inds = img_matches[img1_ind]
        img2s = get_images_indices_from_indices_and_point(ts2, img2_inds, expected_new_center)

        actual_matches_num += 1
        if img1_ind not in prelimdict:
            prelimdict[img1_ind] = {}
        newdi = prelimdict[img1_ind]
        for img2_ind in img2s:
            # Build dictionary here
            if img2_ind not in newdi:
                newdi[img2_ind] = []
            newdi[img2_ind].append(hexgr[i])

    # Generate arguments for PMCC matchingf
    print("Generating PMCC Matching Arguments")
    commandlist = []
    point_matches = []
    for img1_ind in prelimdict.keys():
        commandlist.append((ts1, ts2, template_size, scaling, img1_ind, best_transformations, mfov_centers1, prelimdict, min_corr, max_curvature, max_rod, debug_save_matches, debug_dir))

    # Execute PMCC Matching
    print("Performing PMCC Matching with {} processes".format(processes_num))
    pool = mp.Pool(processes=processes_num)
    results = pool.map(perform_pmcc_unwrapper, commandlist)
    point_matches = [item for sublist in results for item in sublist]

    print("Found {} matches out of possible {} points (on section points: {})".format(len(point_matches), len(hexgr), actual_matches_num))

    # Save the output
    print("Saving output to: {}".format(out_fname))
    out_jsonfile['runtime'] = time.time() - starttime
    out_jsonfile['mesh'] = hexgr
    if targeted_mfov != -1:
        out_jsonfile['mfov1'] = targeted_mfov

    final_point_matches = []
    for pm in point_matches:
        p1, p2, nmesh, match_val = pm
        record = {}
        record['point1'] = p1.tolist()
        record['point2'] = p2.tolist()
        record['isvirtualpoint'] = nmesh
        record['match_val'] = match_val
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
    parser.add_argument('-m', '--mfov', type=int,
                        help='the mfov number of compare (default: all)',
                        default=-1)

    args = parser.parse_args()
    match_layers_pmcc_matching(args.tiles_file1, args.tiles_file2,
                               args.pre_matches_file, args.output_file,
                               targeted_mfov=args.mfov,
                               conf_fname=args.conf_file_name, processes_num=args.threads_num)

if __name__ == '__main__':
    main()
