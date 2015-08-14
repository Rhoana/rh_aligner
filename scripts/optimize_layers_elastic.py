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

cached_radius = None

def compute_restricted_moving_ls_radius(url_optimized_mesh):
    global cached_radius

    # TODO - verify that the radius is dependent only on the mesh (not on the matches)
    if cached_radius is None:
        print "Computing restricted MLS radius"
        min_point_dist_sqr = float("inf")
        for p1idx,p2idx in itertools.combinations(range(len(url_optimized_mesh[0])), 2):
            p1 = url_optimized_mesh[0][p1idx]
            p2 = url_optimized_mesh[0][p2idx]
            cur_dist_sqr = (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])
            min_point_dist_sqr = min(min_point_dist_sqr, cur_dist_sqr)
        cached_radius = 2 * math.sqrt(min_point_dist_sqr)
        print "Restricted MLS radius: {}".format(cached_radius)
        
    return cached_radius


def get_restricted_moving_ls_transform(url_optimized_mesh, bbox):
    # Find the tile bbox with a halo of radius around it
    radius = compute_restricted_moving_ls_radius(url_optimized_mesh)
    bbox_with_halo = list(bbox)
    bbox_with_halo[0] -= radius
    bbox_with_halo[2] -= radius
    bbox_with_halo[1] += radius
    bbox_with_halo[3] += radius

    # filter the matches according to the new bounding box
    matches_str = " ".join(["{0} {1} {2} {3} 1.0".format(m[0][0], m[0][1], m[1][0], m[1][1])
                             for m in zip(url_optimized_mesh[0], url_optimized_mesh[1])
                                 if (bbox_with_halo[0] <= m[0][0] <= bbox_with_halo[1]) and (bbox_with_halo[2] <= m[0][1] <= bbox_with_halo[3])])

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




def save_optimized_meshes(all_tile_urls, optimized_meshes, out_dir):
    for ts_url in all_tile_urls:
        ts_fname = ts_url.replace('file://', '')
        ts_base = os.path.basename(ts_fname)
        out_fname = os.path.join(out_dir, ts_base)
        # read tilespec
        data = None
        with open(ts_fname, 'r') as data_file:
            data = json.load(data_file)

        if len(data) > 0:
            #transform = get_moving_ls_transform(optimized_meshes[ts_url])

            # change the transfromation
            for tile in data:
                # Used for restricting the restricted_moving_ls_transform to a specific bbox
                tile_transform = get_restricted_moving_ls_transform(optimized_meshes[ts_fname], tile["bbox"])
                if "transforms" not in tile.keys():
                    tile["transforms"] = []
                tile["transforms"].append(tile_transform)

            # save the output tile spec
            with open(out_fname, 'w') as outjson:
                json.dump(data, outjson, sort_keys=True, indent=4)
                print('Wrote tilespec to {0}'.format(out_fname))
        else:
            print('Nothing to write for tilespec {}'.format(ts_fname))
        
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
    save_optimized_meshes(all_tile_urls, optimized_meshes, out_dir)
    


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

