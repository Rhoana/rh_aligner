from __future__ import print_function
from rh_aligner.alignment.block_match_3d_multiprocess import match_layers_pmcc_matching
import argparse


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Given two tilespecs of two sections, and a preliminary matches list, generates a grid the image, and performs block matching (with PMCC filtering).')
    parser.add_argument('tiles_file1', metavar='tiles_file1', type=str,
                        help='the first layer json file containing tilespecs')
    parser.add_argument('tiles_file2', metavar='tiles_file2', type=str,
                        help='the second layer json file containing tilespecs')
    parser.add_argument('pre_matches_file', metavar='pre_matches_file', type=str,
                        help='a json file that contains the preliminary matches')
    parser.add_argument('mfov', type=int,
                        help='the mfov number of compare')
    parser.add_argument('-o', '--output_file', type=str,
                        help='an output correspondent_spec file, that will include the matches between the sections (default: ./matches.json)',
                        default='./matches.json')
    parser.add_argument('-c', '--conf_file_name', type=str,
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)
    parser.add_argument('-t', '--threads_num', type=int,
                        help='the number of threads (processes) to use (default: 1)',
                        default=1)

    args = parser.parse_args()
    match_layers_pmcc_matching(args.tiles_file1, args.tiles_file2,
                               args.pre_matches_file, args.output_file,
                               args.mfov,
                               conf_fname=args.conf_file_name, processes_num=args.threads_num)

if __name__ == '__main__':
    main()
