from rh_aligner.stitching.match_sift_features_and_filter_cv2 import match_single_sift_features_and_filter, match_multiple_sift_features_and_filter
import os
import time
import argparse
import re

def wait_after_file(filename, timeout_seconds):
    if timeout_seconds > 0:
        cur_time = time.time()
        mod_time = os.path.getmtime(filename)
        end_wait_time = mod_time + timeout_seconds
        while cur_time < end_wait_time:
            print "Waiting for file: {}".format(filename)
            cur_time = time.time()
            mod_time = os.path.getmtime(filename)
            end_wait_time = mod_time + timeout_seconds
            if cur_time < end_wait_time:
                time.sleep(end_wait_time - cur_time)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over the tilespecs in a file, computing matches for each overlapping tile.')
    parser.add_argument('tiles_file', metavar='tiles_file', type=str,
                        help='the json file of tilespecs')
    parser.add_argument('features_file1', metavar='features_file1', type=str,
                        help='a file that contains the features json file of the first tile (if a single pair is matched) or a list of features json files (if multiple pairs are matched)')
    parser.add_argument('features_file2', metavar='features_file2', type=str,
                        help='a file that contains the features json file of the second tile (if a single pair is matched) or a list of features json files (if multiple pairs are matched)')
    parser.add_argument('--index_pairs', metavar='index_pairs', type=str, nargs='+',
                        help='a colon separated indices of the tiles in the tilespec file that correspond to the feature files that need to be matched. The format is [mfov1_index]_[tile_index]:[mfov2_index]_[tile_index]')
    parser.add_argument('-o', '--output_file', type=str,
                        help='an output file name where the correspondent_spec file will be (if a single pair is matched, default: ./matched_sifts.json) or a list of output files (if multiple pairs are matched, default: ./matched_siftsX.json)',
                        default='./matched_sifts.json')
    parser.add_argument('-c', '--conf_file_name', type=str,
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)
    parser.add_argument('-w', '--wait_time', type=int,
                        help='the time to wait since the last modification date of the features_file (default: None)',
                        default=0)
    parser.add_argument('-t', '--threads_num', type=int,
                        help='the number of threads (processes) to use (default: 1)',
                        default=1)

    args = parser.parse_args()
    print("args:", args)


    if len(args.index_pairs) == 1:
        wait_after_file(args.features_file1, args.wait_time)
        wait_after_file(args.features_file2, args.wait_time)
        
        m = re.match('([0-9]+)_([0-9]+):([0-9]+)_([0-9]+)', args.index_pairs[0])
        index_pair = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
        match_single_sift_features_and_filter(args.tiles_file, args.features_file1, args.features_file2,
                                              args.output_file, index_pair, conf_fname=args.conf_file_name)
    else: # More than one pair

        with open(args.features_file1, 'r') as f_fnames:
            features_files_lst1 = [fname.strip() for fname in f_fnames.readlines()]
        with open(args.features_file2, 'r') as f_fnames:
            features_files_lst2 = [fname.strip() for fname in f_fnames.readlines()]
        for feature_file in zip(features_files_lst1, features_files_lst2):
            wait_after_file(feature_file[0], args.wait_time)
            wait_after_file(feature_file[1], args.wait_time)
        with open(args.output_file, 'r') as f_fnames:
            output_files_lst = [fname.strip() for fname in f_fnames.readlines()]

        index_pairs = []
        for index_pair in args.index_pairs:
            m = re.match('([0-9]+)_([0-9]+):([0-9]+)_([0-9]+)', index_pair)
            index_pairs.append( (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))) )
        match_multiple_sift_features_and_filter(args.tiles_file, features_files_lst1, features_files_lst2,
                                                output_files_lst, index_pairs, conf_fname=args.conf_file_name,
                                                processes_num=args.threads_num)

    print("Done.")

if __name__ == '__main__':
    main()

