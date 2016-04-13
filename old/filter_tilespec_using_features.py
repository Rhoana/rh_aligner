# Filters a given tilespec using the number of features in each tile
# if a tile has less features than some threshold, that tile is filtered out
import sys
import json
import os
import h5py
import utils
import argparse

def wait_for_feature_files(features_fname, wait_time):
    utils.wait_after_file(features_fname, wait_time)

def check_tile_sift_file(sift_file, threshold):
    # load the h5py file
    try:
        with h5py.File(sift_file, 'r') as data:
            descs_count = data['descs'].shape[0]

        return descs_count >= threshold
    except:
        print "Error when parsing: {}".format(sift_file)
        raise

def get_tilespec_sift_files(ts, ts_sifts_dir):
    # Fetch all the sift files, per mfov directory
    per_mfov_sift_files = {}
    len_ts_sifts_dir = len(ts_sifts_dir)
    for entry in os.walk(ts_sifts_dir):
        # if the directory is a mfov directory
        if len(entry[0]) > len_ts_sifts_dir:
            mfov_str = entry[0][len_ts_sifts_dir:].replace('/', '')
            if mfov_str.isdigit():
                per_mfov_sift_files[mfov_str] = [os.path.join(entry[0], f) for f in sorted(entry[2])]

    corresponding_sift_files = []
    # match for each tile in the tilespec the corresponding file
    for tile_ts in ts:
        mfov_str = str(tile_ts["mfov"]).zfill(6)
        tile_index = tile_ts["tile_index"]
        # TODO - the following assumes that sift features were computed for all tiles and only for the tiles
        feature_file = per_mfov_sift_files[mfov_str][tile_index - 1]

        # The feature file should have the same name as the following: [tilespec base filename]_sifts_[img filename].json/.hdf5/.h5py
        assert(os.path.basename(os.path.splitext(tile_ts["mipmapLevels"]["0"]["imageUrl"])[0]) in feature_file)
        corresponding_sift_files.append(feature_file)

    return corresponding_sift_files

def filter_tilespec_using_features(in_ts_fname, out_ts_fname, ts_sifts_dir, wait_time=30, conf_fname=None):
    params = utils.conf_from_file(conf_fname, 'FilterTilespecUsingFeatures')
    if params is None:
        params = {}
    sift_num_threshold = params.get("sift_num_threshold", 1000)

    ts = utils.load_tilespecs(in_ts_fname)

    # Find the corresponding sift files
    ts_sift_files = get_tilespec_sift_files(ts, ts_sifts_dir)

    # filter the tilespec
    out_ts = []
    for tile_ts, sift_file in zip(ts, ts_sift_files):
        sift_file_path = os.path.join(ts_sifts_dir, sift_file)
        wait_for_feature_files(sift_file_path, wait_time)
        if check_tile_sift_file(sift_file_path, sift_num_threshold):
            out_ts.append(tile_ts)

    # Save the new tilespec
    with open(out_ts_fname, 'w') as out:
        json.dump(out_ts, out, sort_keys=True, indent=4)

def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Given a tilespec file, its corresponding sift files directory, and a threshold, \
        filters out all the tiles that have less sift features than the given threshold.')
    parser.add_argument('tiles_fname', metavar='tiles_json', type=str,
                        help='a tile_spec file that contains the tile images to filter, in json format')
    parser.add_argument('sifts_dir', metavar='sifts_dir', type=str,
                        help='the directory that contains a per-mfov hdf5 files of the sift features')
    parser.add_argument('-o', '--output_fname', type=str,
                        help='output tile_spec with only the relevant tiles')
    parser.add_argument('-c', '--conf_file_name', type=str,
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)
    parser.add_argument('-w', '--wait_time', type=int,
                        help='the time to wait since the last modification date of the features_file (default: None)',
                        default=0)
    #parser.add_argument('-t', '--threads_num', type=int,
    #                    help='the number of threads (processes) to use (default: 1)',
    #                    default=1)


    args = parser.parse_args()
    print args

    filter_tilespec_using_features(args.tiles_fname, args.output_fname, args.sifts_dir, args.wait_time, args.conf_file_name)

if __name__ == '__main__':
    main()

