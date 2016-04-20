from __future__ import print_function
from rh_aligner.alignment.pre_match_3d_incremental import match_layers_sift_features
import argparse

def main():
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
