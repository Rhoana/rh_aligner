import sys
import os
import glob
import argparse
from subprocess import call
import utils


def render_2d(tiles_fname, output_fname, width, jar_file, threads_num=None):

    tiles_url = utils.path2url(tiles_fname)

    threads_str = ""
    if threads_num is not None:
        threads_str = "--threads {}".format(threads_num)

    width_str = "--width {}".format(width)
    if width == -1:
        width_str = "--fullImage"

    java_cmd = 'java -Xmx32g -XX:ParallelGCThreads=1 -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.Render --out {1} {2} \
        {3} --hide --url {4}'.format(\
            jar_file, output_fname, width_str, threads_str, tiles_url)
    utils.execute_shell_command(java_cmd)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Renders a single tilespec file (with all its tiles)')
    parser.add_argument('tiles_file', metavar='tiles_file', type=str,
                        help='a json tilespecs file that need to be rendered')
    parser.add_argument('-o', '--output_fname', type=str, 
                        help='an output file (default: ./tilespec_fname.tif)',
                        default=None)
    parser.add_argument('-j', '--jar_file', type=str,
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')
    parser.add_argument('-t', '--threads_num', type=int, 
                        help='the number of threads to use (default: 1)',
                        default=1)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-q', '--quality', type=str, choices=['full', 'default', 'fast', 'veryfast'],
                        help='sets the output quality and resoultion')
    group.add_argument('-w', '--width', type=int, 
                        help='set the width of the rendered images')

    args = parser.parse_args()

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

    output_fname = args.output_fname
    if args.output_fname is None:
        output_fname = os.path.basename(args.tiles_file) + '.tif'

    render_2d(args.tiles_file, output_fname, width, args.jar_file, args.threads_num)

if __name__ == '__main__':
    main()

