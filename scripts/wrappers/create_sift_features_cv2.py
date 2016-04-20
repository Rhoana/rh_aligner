from rh_aligner.stitching.create_sift_features_cv2 import create_multiple_sift_features
import argparse



def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over a directory that contains json files, \
        and creates the sift features of each file. \
        The output is either in the same directory or in a different, user-provided, directory \
        (in either case, we use a different file name). \
        The order and number of the indices must match the output files.')
    parser.add_argument('tiles_fname', metavar='tiles_json', type=str, 
                        help='a tile_spec file that contains the images to create sift features for, in json format')
    parser.add_argument('-i', '--indices', type=int, nargs='+', 
                        help='the indices of the tiles in the tilespec that needs to be computed')
    parser.add_argument('-o', '--output_files', type=str, nargs='+', 
                        help='output feature_spec files list, each will include the sift features for the corresponding tile')
    parser.add_argument('-c', '--conf_file_name', type=str, 
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)


    args = parser.parse_args()
    print args

    assert(len(args.output_files) == len(args.indices))
    create_multiple_sift_features(args.tiles_fname, args.output_files, args.indices, conf_fname=args.conf_file_name)

if __name__ == '__main__':
    main()

