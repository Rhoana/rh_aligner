import sys
import os
import glob
import argparse
from subprocess import call
import utils


def render_tiles_2d(tiles_fname, output_dir, tile_size, jar_file, threads_num=None):

    tiles_url = utils.path2url(tiles_fname)

    threads_str = ""
    if threads_num is not None:
        threads_str = "--threads {}".format(threads_num)

    java_cmd = 'java -Xmx32g -XX:ParallelGCThreads=1 -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.RenderTiles --targetDir {1} --tileSize {2} \
        {3} --url {4}'.format(\
            jar_file, output_dir, tile_size, threads_str, tiles_url)
    utils.execute_shell_command(java_cmd)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Renders a single tilespec file (with all its tiles) into a tiled image')
    parser.add_argument('tiles_file', metavar='tiles_file', type=str, nargs='+',
                        help='a json tilespecs file that need to be rendered')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='The directory where the tiled images will be stored (default: ./output_tiles)',
                        default='./output_tiles')
    parser.add_argument('-j', '--jar_file', type=str,
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')
    parser.add_argument('-t', '--threads_num', type=int, 
                        help='the number of threads to use (default: 1)',
                        default=1)
    parser.add_argument('-s', '--tile_size', type=int, 
                        help='the size (square side) of each tile (default: 512)',
                        default=512)

    args = parser.parse_args()


    utils.create_dir(args.output_dir)

    render_tiles_2d(args.tiles_file, args.output_dir, args.tile_size, args.jar_file, args.threads_num)

if __name__ == '__main__':
    main()

