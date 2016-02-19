import sys
import os
import glob
import argparse
from subprocess import call
from bounding_box import BoundingBox
import json
import itertools
import utils
import export_mesh
from optimize_mesh import optimize_meshes
import math
import numpy as np
from scipy import spatial
import multiprocessing as mp

SAMPLED_POINTS_NUM = 50

def compute_restricted_moving_ls_radius(url_optimized_mesh0, points_tree):

    # TODO - verify that the radius is dependent only on the mesh (not on the matches)
    print "Computing restricted MLS radius"

    # Sample SAMPLED_POINTS_NUM points to find the closest neighbors to these points
    sampled_points_indices = np.random.choice(url_optimized_mesh0.shape[0], SAMPLED_POINTS_NUM, replace=False)
    sampled_points = url_optimized_mesh0[np.array(sampled_points_indices)]

    # Find the minimal distance between the sampled points to any other point, by finding the closest point to each point
    # and take the minimum among the distances
    distances, _ = points_tree.query(sampled_points, 2)
    min_point_dist = np.min(distances[:,1])
    radius = 2 * min_point_dist
    print "Restricted MLS radius: {}".format(radius)

    return radius


def get_restricted_moving_ls_transform(url_optimized_mesh, bbox, points_tree, radius):
    # Find the tile bbox with a halo of radius around it
    bbox_with_halo = list(bbox)
    bbox_with_halo[0] -= radius
    bbox_with_halo[2] -= radius
    bbox_with_halo[1] += radius
    bbox_with_halo[3] += radius

    # filter the matches according to the new bounding box
    # (first pre-filter entire mesh using a radius of "diagonal + 2*RMLS_radius + 1) around the top-left point)
    top_left = np.array([bbox[0], bbox[2]])
    bottom_right = np.array([bbox[1], bbox[3]])
    pre_filtered_indices = points_tree.query_ball_point(top_left, np.linalg.norm(bottom_right - top_left) + 2 * radius + 1)
    #print bbox_with_halo, "with filtered_indices:", pre_filtered_indices

    if len(pre_filtered_indices) == 0:
        print "Could not find any mesh points in bbox {}, skipping the tile"
        return None

    filtered_matches = [m for m in zip(url_optimized_mesh[0][np.array(pre_filtered_indices)], url_optimized_mesh[1][np.array(pre_filtered_indices)])
                                if (bbox_with_halo[0] <= m[0][0] <= bbox_with_halo[1]) and (bbox_with_halo[2] <= m[0][1] <= bbox_with_halo[3])]

    if len(filtered_matches) == 0:
        print "Could not find any mesh points in bbox {}, skipping the tile"
        return None

    matches_str = " ".join(["{0} {1} {2} {3} 1.0".format(m[0][0], m[0][1], m[1][0], m[1][1])
                            for m in filtered_matches])

    # print bbox_with_halo, "with pre_filtered_indices len:", len(pre_filtered_indices), "with matches_str:", matches_str
    # create the tile transformation
    transform = {
            "className" : "mpicbg.trakem2.transform.RestrictedMovingLeastSquaresTransform2",
            "dataString" : "affine 2 2.0 {0} {1}".format(radius, matches_str)
        }
    return transform

def get_moving_ls_transform(url_optimized_mesh):
    all_matches_str = " ".join(["{0} {1} {2} {3} 1.0".format(m[0][0], m[0][1], m[1][0], m[1][1]) for m in zip(url_optimized_mesh[0], url_optimized_mesh[1])])
# change the transfromation
    transform = {
            "className" : "mpicbg.trakem2.transform.MovingLeastSquaresTransform2",
            "dataString" : "affine 2 2.0 {0}".format(all_matches_str)
        }
    return transform


def save_json_file(out_fname, data):
    with open(out_fname, 'w') as outjson:
        json.dump(data, outjson, sort_keys=True, indent=4)
        print('Wrote tilespec to {0}'.format(out_fname))
        sys.stdout.flush()

def save_optimized_mesh(ts_fname, url_optimized_mesh, out_dir):
    # pre-compute the restricted MLS radius for that section
    print "Working on:", ts_fname

    points_tree = spatial.KDTree(url_optimized_mesh[0])
    radius = compute_restricted_moving_ls_radius(url_optimized_mesh[0], points_tree)

    ts_base = os.path.basename(ts_fname)
    out_fname = os.path.join(out_dir, ts_base)
    # read tilespec
    data = None
    with open(ts_fname, 'r') as data_file:
        data = json.load(data_file)

    if len(data) > 0:
        tiles_to_remove = []
        # change the transfromation
        for tile_index, tile in enumerate(data):
            # Used for restricting the restricted_moving_ls_transform to a specific bbox
            tile_transform = get_restricted_moving_ls_transform(url_optimized_mesh, tile["bbox"], points_tree, radius)
            if tile_transform is None:
                tiles_to_remove.append(tile_index)
            else:
                tile.get("transforms", []).append(tile_transform)

        for tile_index in sorted(tiles_to_remove, reverse=True):
            print "Removing tile {} from {}".format(data[tile_index]["mipmapLevels"]["0"]["imageUrl"], out_fname)
            del data[tile_index]

        # save the output tile spec
        save_json_file(out_fname, data)
    else:
        print('Nothing to write for tilespec {}'.format(ts_fname))
        sys.stdout.flush()


def save_optimized_meshes(all_tile_urls, optimized_meshes, out_dir, processes_num=1):
    # Do the actual multiprocessed save
    pool = mp.Pool(processes=processes_num)
    print("Using {} processes to save the output jsons".format(processes_num))

    for ts_url in all_tile_urls:
        ts_fname = ts_url.replace('file://', '')
        #res = pool.apply_async(save_optimized_mesh, (ts_fname, optimized_meshes[ts_fname], out_dir))
        save_optimized_mesh(ts_fname, optimized_meshes[ts_fname], out_dir)
    pool.close()
    pool.join()
        
def read_ts_layers(tile_files):
    tsfile_to_layerid = {}

    print "Reading tilespec files"

    # TODO - make sure its not a json files list
    actual_tile_urls = []
    with open(tile_files[0], 'r') as f:
        actual_tile_urls = [line.strip('\n') for line in f.readlines()]
    
    for url in actual_tile_urls:
        file_name = url.replace('file://', '')
        layerid = utils.read_layer_from_file(file_name)
        tsfile = os.path.basename(url)
        tsfile_to_layerid[tsfile] = layerid

    return tsfile_to_layerid, actual_tile_urls
    


def optimize_layers_elastic(tile_files, corr_files, out_dir, max_layer_distance, conf=None, skip_layers=None, threads_num=4):

    tsfile_to_layerid, all_tile_urls = read_ts_layers(tile_files)

    # TODO: the tile_files order should imply the order of sections

    # TODO - make sure its not a json files list
    actual_corr_files = []
    with open(corr_files[0], 'r') as f:
        actual_corr_files = [line.replace('file://', '').strip('\n') for line in f.readlines()]

    conf_dict = {}
    if conf is not None:
        with open(conf, 'r') as f:
            conf_dict = json.load(f)["OptimizeLayersElastic"]

    print(actual_corr_files)
    # Create a per-layer optimized mesh
    optimized_meshes = optimize_meshes(actual_corr_files, conf_dict)
    
    # Save the output
    utils.create_dir(out_dir)
    save_optimized_meshes(all_tile_urls, optimized_meshes, out_dir, threads_num)

    print("Done.")
    


def main():
    print(sys.argv)

    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over the tilespecs in a file, computing matches for each overlapping tile.')
    parser.add_argument('--tile_files', metavar='tile_files', type=str, nargs='+', required=True,
                        help='the list of tile spec files to align')
    parser.add_argument('--corr_files', metavar='corr_files', type=str, nargs='+', required=True,
                        help='the list of corr spec files that contain the matched layers')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='an output directory that will include the aligned sections tiles (default: .)',
                        default='./')
    parser.add_argument('-c', '--conf_file_name', type=str, 
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)
    parser.add_argument('-t', '--threads_num', type=int, 
                        help='the number of threads to use (default: 1)',
                        default=1)
    parser.add_argument('-s', '--skip_layers', type=str, 
                        help='the range of layers (sections) that will not be processed e.g., "2,3,9-11,18" (default: no skipped sections)',
                        default=None)
    parser.add_argument('-d', '--max_layer_distance', type=int, 
                        help='the largest distance between two layers to be matched (default: 1)',
                        default=1)


    args = parser.parse_args()

    print "tile_files: {0}".format(args.tile_files)
    print "corr_files: {0}".format(args.corr_files)

    optimize_layers_elastic(args.tile_files, args.corr_files,
        args.output_dir, args.max_layer_distance,
        conf=args.conf_file_name, 
        skip_layers=args.skip_layers, threads_num=args.threads_num)

if __name__ == '__main__':
    main()

