from __future__ import print_function
import sys
import cv2
from rh_renderer.tilespec_affine_renderer import TilespecAffineRenderer
import argparse
import utils
from bounding_box import BoundingBox
from rh_renderer import models
import numpy as np
import json
from scipy.spatial import distance
from scipy import spatial
import time
import subprocess
import os


def get_closest_index_to_point(point, centerstree):
    distanc, closest_index = centerstree.query(point)
    return closest_index

def is_point_in_img(tile_ts, point):
    """Returns True if the given point lies inside the image as denoted by the given tile_tilespec"""
    # TODO - instead of checking inside the bbox, need to check inside the polygon after transformation
    img_bbox = tile_ts["bbox"]

    if point[0] > img_bbox[0] and point[1] > img_bbox[2] and \
       point[0] < img_bbox[1] and point[1] < img_bbox[3]:
        return True
    return False

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

def get_best_transformations(pre_mfov_matches, tiles_fname1, tiles_fname2, mfov_centers1, mfov_centers2, sorted_mfovs1, sorted_mfovs2):
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
            mfov_num2 = sorted_mfovs1[closest_idx]
            transforms[mfov_num2] = reversed_transform

    else:
        for m in pre_mfov_matches["matches"]:
            transforms[m["mfov1"]] = m["transformation"]["matrix"]

    # Add the "transformations" of all the mfovs w/o a direct mapping (using their neighbors)
    estimated_transforms = {}
    trans_keys = transforms.keys()
    for m in sorted_mfovs1:
        if m not in transforms.keys():
            # Need to find a more intelligent way to do this, but this suffices for now
            # Uses transformation of another mfov (should be changed to the closest, unvisited mfov)
            mfov_center = mfov_centers1[m]
            closest_mfov_idx = np.argmin([distance.euclidean(mfov_center, mfov_centers1[mfovk]) for mfovk in trans_keys])
            estimated_transforms[m] = transforms[trans_keys[closest_mfov_idx]]

    transforms.update(estimated_transforms)

    return transforms

def find_best_mfov_transformation(mfov, best_transformations, mfov_centers):
    """Returns a matrix that represnets the best transformation for a given mfov to the other section"""
    if mfov in best_transformations.keys():
        return best_transformations[mfov]
    # Need to find a more intelligent way to do this, but this suffices for now
    # Uses transformation of another mfov (should be changed to the closest, unvisited mfov)
    mfov_center = mfov_centers[mfov]
    trans_keys = best_transformations.keys()
    closest_mfov_idx = np.argmin([distance.euclidean(mfov_center, mfov_centers[mfovk]) for mfovk in trans_keys])
    # ** Should we add this transformation? maybe not, because we don't want to get a different result when a few
    # ** missing mfov matches occur and the "best transformation" can change when the centers are updated
    return best_transformations[trans_keys[closest_mfov_idx]]

def get_tile_centers_from_json(ts):
    tiles_centers = []
    for tile in ts:
        center_x = (tile["bbox"][0] + tile["bbox"][1]) / 2.0
        center_y = (tile["bbox"][2] + tile["bbox"][3]) / 2.0
        tiles_centers.append(np.array([center_x, center_y]))
    return tiles_centers

def save_animated_gif(img1, start_point1, img2, start_point2, output_fname):
    end_point1 = start_point1 + np.array([img1.shape[1], img1.shape[0]])
    end_point2 = start_point2 + np.array([img2.shape[1], img2.shape[0]])

    # merge the 2 images into a single image using 2 channels (red and green)
    start_point = np.minimum(start_point1, start_point2)
    end_point = np.maximum(end_point1, end_point2)

    out_shape = ((end_point - start_point)[1], (end_point - start_point)[0])
    #img_out = np.zeros(((end_point - start_point)[1], (end_point - start_point)[0], 3), dtype=np.uint8)
    #img_out[start_point1[1] - start_point[1]:img1.shape[0] + start_point1[1] - start_point[1],
    #        start_point1[0] - start_point[0]:img1.shape[1] + start_point1[0] - start_point[0],
    #        1] = img1
    #img_out[start_point2[1] - start_point[1]:img2.shape[0] + start_point2[1] - start_point[1],
    #        start_point2[0] - start_point[0]:img2.shape[1] + start_point2[0] - start_point[0],
    #        2] = img2
    img_out = np.zeros(out_shape, dtype=np.uint8)
    img_out[start_point1[1] - start_point[1]:img1.shape[0] + start_point1[1] - start_point[1],
            start_point1[0] - start_point[0]:img1.shape[1] + start_point1[0] - start_point[0]] = img1
    img1_output_fname = "{}_1.jpg".format(output_fname)
    cv2.imwrite(img1_output_fname, img_out)
    img_out = np.zeros(out_shape, dtype=np.uint8)
    img_out[start_point2[1] - start_point[1]:img2.shape[0] + start_point2[1] - start_point[1],
            start_point2[0] - start_point[0]:img2.shape[1] + start_point2[0] - start_point[0]] += img2
    img2_output_fname = "{}_2.jpg".format(output_fname)
    cv2.imwrite(img2_output_fname, img_out)

    gif_output_fname = "{}.gif".format(output_fname)

    #Combine into a GIF using ImageMagick's "convert"-command (called using subprocess.call()):
    convertexepath = "convert"
    convertcommand = [convertexepath, "-delay", "10", "-size", str(out_shape[1]) + "x" + str(out_shape[0]), img1_output_fname, img2_output_fname, gif_output_fname]
    subprocess.call(convertcommand)



def view_pre_pmcc_mfov(pre_matches_fname, targeted_mfov, output_fname, scale):

    # Load the preliminary matches
    with open(pre_matches_fname, 'r') as data_matches:
        mfov_pre_matches = json.load(data_matches)
    if len(mfov_pre_matches["matches"]) == 0:
        print("No matches were found in pre-matching, aborting")
        return

    tiles_fname1 = mfov_pre_matches["tilespec1"]
    tiles_fname2 = mfov_pre_matches["tilespec2"]

    # Read the tilespecs
    ts1 = utils.load_tilespecs(tiles_fname1)
    ts2 = utils.load_tilespecs(tiles_fname2)
    indexed_ts1 = utils.index_tilespec(ts1)
    indexed_ts2 = utils.index_tilespec(ts2)

    sorted_mfovs1 = sorted(indexed_ts1.keys())
    sorted_mfovs2 = sorted(indexed_ts2.keys())

    # Get the tiles centers for each section
    tile_centers1 = get_tile_centers_from_json(ts1)
    tile_centers1tree = spatial.KDTree(tile_centers1)
    tile_centers2 = get_tile_centers_from_json(ts2)
    tile_centers2tree = spatial.KDTree(tile_centers2)
    mfov_centers1 = get_mfov_centers_from_json(indexed_ts1)
    mfov_centers2 = get_mfov_centers_from_json(indexed_ts2)

    best_transformations = get_best_transformations(mfov_pre_matches, tiles_fname1, tiles_fname2, mfov_centers1, mfov_centers2, sorted_mfovs1, sorted_mfovs2)

    mfov_tiles = indexed_ts1[targeted_mfov]
    tiles_boundaries1 = []
    for tile in mfov_tiles.values():
        p1 = np.array([tile["bbox"][0], tile["bbox"][2]])
        p2 = np.array([tile["bbox"][0], tile["bbox"][3]])
        p3 = np.array([tile["bbox"][1], tile["bbox"][2]])
        p4 = np.array([tile["bbox"][1], tile["bbox"][3]])
        tiles_boundaries1.extend([p1, p2, p3, p4])

    # Create the (lazy) renderers for the two sections
    img1_renderer = TilespecAffineRenderer(indexed_ts1[targeted_mfov].values())

    # Use the mfov exepected transform (from section 1 to section 2) to transform img1
    img1_to_img2_transform = np.array(find_best_mfov_transformation(targeted_mfov, best_transformations, mfov_centers1)[:2])
    img1_renderer.add_transformation(img1_to_img2_transform)

    # Find the relevant tiles from section 2
    section2_tiles_indices = set()
    
    for p in tiles_boundaries1:
        # Find the tile image where the point from the hexagonal is in the first section
        img1_ind = get_closest_index_to_point(p, tile_centers1tree)
        #print(img1_ind)
        if img1_ind is None:
            continue
        if ts1[img1_ind]["mfov"] != targeted_mfov:
            continue
        if not is_point_in_img(ts1[img1_ind], p):
            continue

        img1_point = p

        # Find the point on img2
        img1_point_on_img2 = np.dot(img1_to_img2_transform[:2,:2], img1_point) + img1_to_img2_transform[:2,2]
        
        # Find the tile that is closest to that point
        img2_ind = get_closest_index_to_point(img1_point_on_img2, tile_centers2tree)
        #print("img1_ind {}, img2_ind {}".format(img1_ind, img2_ind))
        section2_tiles_indices.add(img2_ind)

    print("section2 tiles (#tiles={}): {}".format(len(section2_tiles_indices), section2_tiles_indices))

    # Scale down the rendered images
    scale_transformation = np.array([
                                [ scale, 0., 0. ],
                                [ 0., scale, 0. ]
                            ])
    img1_renderer.add_transformation(scale_transformation)

    img2_renderer = TilespecAffineRenderer([ts2[tile_index] for tile_index in section2_tiles_indices])
    img2_renderer.add_transformation(scale_transformation)


    # render the images
    start_time = time.time()
    img1, start_point1 = img1_renderer.render()
    print("image 1 rendered in {} seconds".format(time.time() - start_time))
    start_time = time.time()
    img2, start_point2 = img2_renderer.render()
    print("image 2 rendered in {} seconds".format(time.time() - start_time))


    end_point1 = start_point1 + np.array([img1.shape[1], img1.shape[0]])
    end_point2 = start_point2 + np.array([img2.shape[1], img2.shape[0]])

    # merge the 2 images into a single image using 2 channels (red and green)
    start_point = np.minimum(start_point1, start_point2)
    end_point = np.maximum(end_point1, end_point2)

#    out_shape = ((end_point - start_point)[1], (end_point - start_point)[0])
#    #img_out = np.zeros(((end_point - start_point)[1], (end_point - start_point)[0], 3), dtype=np.uint8)
#    #img_out[start_point1[1] - start_point[1]:img1.shape[0] + start_point1[1] - start_point[1],
#    #        start_point1[0] - start_point[0]:img1.shape[1] + start_point1[0] - start_point[0],
#    #        1] = img1
#    #img_out[start_point2[1] - start_point[1]:img2.shape[0] + start_point2[1] - start_point[1],
#    #        start_point2[0] - start_point[0]:img2.shape[1] + start_point2[0] - start_point[0],
#    #        2] = img2
#    img_out = np.zeros(((end_point - start_point)[1], (end_point - start_point)[0]), dtype=np.uint8)
#    img_out[start_point1[1] - start_point[1]:img1.shape[0] + start_point1[1] - start_point[1],
#            start_point1[0] - start_point[0]:img1.shape[1] + start_point1[0] - start_point[0]] = img1
#    img_out[start_point2[1] - start_point[1]:img2.shape[0] + start_point2[1] - start_point[1],
#            start_point2[0] - start_point[0]:img2.shape[1] + start_point2[0] - start_point[0]] -= img2
#
#    cv2.imwrite(output_fname, img_out)
    save_animated_gif(img1, start_point1, img2, start_point2, output_fname)
    die

def main():
    # print(sys.argv)
    # Command line parser
    parser = argparse.ArgumentParser(description='Given a between slices pre-match json file, and an mfov number, renders both the mfov and the overlapping area using different color.')
    parser.add_argument('pre_matches_file', metavar='pre_matches_file', type=str,
                        help='a json file that contains the preliminary matches')
    parser.add_argument('mfov', type=int,
                        help='the mfov number of compare')
    parser.add_argument('-o', '--output_file', type=str,
                        help='an output png file (default: ./output.png)',
                        default='./output.png')
    parser.add_argument('-s', '--scale', type=float,
                        help='output scale (default: 0.1)',
                        default=0.1)

    args = parser.parse_args()
    view_pre_pmcc_mfov(args.pre_matches_file, args.mfov, args.output_file, args.scale)

if __name__ == '__main__':
    main()
