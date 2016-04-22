from rh_aligner.alignment.normalize_coordinates import normalize_coordinates
import argparse


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Given a list of tilespec file names, normalizes all the tilepecs to a single coordinate system starting from (0,0).')
    parser.add_argument('tile_files_or_dirs', metavar='tile_files_or_dirs', type=str, nargs='+',
                        help='a list of json files that need to be normalized or a directories of json files')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='an output directory (default: ./after_norm)',
                        default='./after_norm')

    args = parser.parse_args()

    normalize_coordinates(args.tile_files_or_dirs, args.output_dir)

if __name__ == '__main__':
    main()

