from rh_aligner.alignment.optimize_layers_elastic import optimize_layers_elastic
import sys
import argparse


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

