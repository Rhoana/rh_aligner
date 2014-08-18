import sys
import os
import glob
import argparse
from subprocess import call
import utils


def render_3d(tile_fnames_or_dir, output_dir, from_layer, to_layer, width, jar_file, threads_num=1):

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
        print "No files for rendering found. Exiting."
        return

    print "Rendering {0} files".format(all_files)

    files_urls = []
    for file_name in all_files:
        tiles_url = utils.path2url(file_name)
        files_urls.append(tiles_url)

    java_cmd = 'java -Xmx4g -XX:ParallelGCThreads=1 -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.Render3D --targetDir {1} --width {2} \
        --threads {3} --fromLayer {4} --toLayer {5} --hide {6}'.format(\
            jar_file, output_dir, width, threads_num, from_layer, to_layer, ' '.join(files_urls))
    utils.execute_shell_command(java_cmd)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Given a list of tilespec file names, normalizes all the tilepecs to a single coordinate system starting from (0,0).')
    parser.add_argument('tile_files_or_dirs', metavar='tile_files_or_dirs', type=str, nargs='+',
                        help='a list of json files that need to be normalized or a directories of json files')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='an output directory (default: ./after_render)',
                        default='./after_render')
    parser.add_argument('-j', '--jar_file', type=str,
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')
    parser.add_argument('-t', '--threads_num', type=int, 
                        help='the number of threads to use (default: 1)',
                        default=1)
    parser.add_argument('--from_layer', type=int, 
                        help='the layer to start from (inclusive, default: the first layer in the data)',
                        default=-1)
    parser.add_argument('--to_layer', type=int, 
                        help='the last layer to render (inclusive, default: the last layer in the data)',
                        default=-1)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-q', '--quality', type=str, choices=['full', 'default', 'fast', 'veryfast'],
                        help='sets the output quality and resoultion')
    group.add_argument('-w', '--width', type=int, 
                        help='set the width of the rendered images')

    args = parser.parse_args()

    utils.create_dir(args.output_dir)

    width = 4000 # default width
    if not args.width is None:
        width = args.width
    elif not args.quality is None:
        if args.quality == 'full':
            width = -1 # full image width
        elif args.quality == 'fast':
            width = 1000
        elif args.quality == 'veryfast':
            width = 100

    print "args: from_layer {0}, to_layer {1}".format(args.from_layer, args.to_layer)
    render_3d(args.tile_files_or_dirs, args.output_dir, args.from_layer, args.to_layer, width, args.jar_file, args.threads_num)

if __name__ == '__main__':
    main()

