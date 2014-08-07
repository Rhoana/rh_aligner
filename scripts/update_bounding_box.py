import sys
import os
import glob
import argparse
from subprocess import call
import utils


# single file bounding box updater
def update_bounding_box(tiles_fname, output_dir, jar_file, threads_num=1):
    tiles_url = utils.path2url(tiles_fname)
    java_cmd = 'java -Xmx3g -XX:ParallelGCThreads=1 -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.UpdateBoundingBox --threads {1} --targetDir {2} {3}'.format(\
        jar_file, threads_num, output_dir, tiles_url)
    utils.execute_shell_command(java_cmd)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over a given directory or json files, and computes the tiles new bounding boxes.')
    parser.add_argument('tile_files_or_dirs', metavar='tile_files_or_dirs', type=str, nargs='+',
                        help='a list of json files that need to be updated or a directories of json files')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='an output directory (default: ./after_bbox)',
                        default='./after_bbox')
    parser.add_argument('-j', '--jar_file', type=str,
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')
    parser.add_argument('-t', '--threads_num', type=int, 
                        help='the number of threads to use (default: 1)',
                        default=1)


    args = parser.parse_args()

    print args.tile_files_or_dirs

    utils.create_dir(args.output_dir)


    for file_or_dir in args.tile_files_or_dirs:
        if not os.path.exists(file_or_dir):
            print "{0} does not exist (file/directory), skipping".format(file_or_dir)
            continue

        if os.path.isdir(file_or_dir):
            actual_files = glob.glob(os.path.join(file_or_dir, '*.json'))
            for file_name in actual_files:
                update_bounding_box(file_name, args.output_dir, args.jar_file, args.threads_num)
        else:
            update_bounding_box(file_or_dir, args.output_dir, args.jar_file, args.threads_num)

if __name__ == '__main__':
    main()

