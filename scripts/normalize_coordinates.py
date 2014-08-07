import sys
import os
import glob
import argparse
from subprocess import call
import utils


def normalize_coordinates(tile_fnames_or_dir, output_dir, jar_file):

    all_files = []

    for file_or_dir in tile_fnames_or_dir:
        if not os.path.exists(file_or_dir):
            print "{0} does not exist (file/directory), skipping".format(file_or_dir)
            continue

        if os.path.isdir(file_or_dir):
            actual_dir_files = glob.glob(os.path.join(file_or_dir, '*.json'))
            all_files.extend(actual_dir_files)
        else:
            all_files.append(file_or_dir)

    if len(all_files) == 0:
        print "No files for normalization found. Exiting."
        return

    print "Normalizing coordinates of {0} files".format(all_files)

    files_urls = []
    for file_name in all_files:
        tiles_url = utils.path2url(file_name)
        files_urls.append(tiles_url)

    java_cmd = 'java -Xmx2g -XX:ParallelGCThreads=1 -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.NormalizeCoordinates --targetDir {1} {2}'.format(\
        jar_file, output_dir, ' '.join(files_urls))
    utils.execute_shell_command(java_cmd)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Given a list of tilespec file names, normalizes all the tilepecs to a single coordinate system starting from (0,0).')
    parser.add_argument('tile_files_or_dirs', metavar='tile_files_or_dirs', type=str, nargs='+',
                        help='a list of json files that need to be normalized or a directories of json files')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='an output directory (default: ./after_norm)',
                        default='./after_norm')
    parser.add_argument('-j', '--jar_file', type=str,
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')

    args = parser.parse_args()

    utils.create_dir(args.output_dir)

    normalize_coordinates(args.tile_files_or_dirs, args.output_dir, args.jar_file)

if __name__ == '__main__':
    main()

