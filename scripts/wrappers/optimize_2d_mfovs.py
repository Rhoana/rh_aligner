from rh_aligner.stitching.optimize_2d_mfovs import optimize_2d_mfovs
import argparse

if __name__ == '__main__':
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over a directory that contains matched points in json files, \
        optimizes these matches into a per-tile transformation, and saves a tilespec json file with these transformations.')
    parser.add_argument('tiles_fname', metavar='tiles_json', type=str,
                        help='a tile_spec file that contains the images to create sift features for, in json format')
    parser.add_argument('match_files_list', metavar='match_files_list', type=str,
                        help="a txt file containg a list of all the match files")
    parser.add_argument('-o', '--output_file', type=str,
                        help='an output tile_spec file, that will include the rotations for all tiles (default: ./output.json)',
                        default='./output.json')
    parser.add_argument('-c', '--conf_file_name', type=str,
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)

    args = parser.parse_args()

    optimize_2d_mfovs(args.tiles_fname, args.match_files_list, args.output_file, conf_fname=args.conf_file_name)
