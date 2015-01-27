import sys
import os
import glob
import argparse
from subprocess import call
import utils


def render_tiles_2d(tiles_fname, output_dir, tile_size, output_type, jar_file, output_pattern=None, blend_type=None, threads_num=None):

    tiles_url = utils.path2url(tiles_fname)

    threads_str = ""
    if threads_num is not None:
        threads_str = "--threads {}".format(threads_num)

    blend_str = ""
    if blend_type is not None:
        blend_str = "--blendType {}".format(blend_type)

    pattern_str = ""
    if output_pattern is not None:
        pattern_str = "--outputNamePattern {}".format(output_pattern)


    java_cmd = 'java -Xmx12g -XX:ParallelGCThreads=1 -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.RenderTiles --targetDir {1} --tileSize {2} \
        {3} {4} --outputType {5} {6} --url {7}'.format(\
            jar_file, output_dir, tile_size, threads_str, blend_str, output_type, pattern_str, tiles_url)
    utils.execute_shell_command(java_cmd)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Renders a single tilespec file (with all its tiles) into a tiled image')
    parser.add_argument('tiles_file', metavar='tiles_file', type=str,
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
    parser.add_argument('-b', '--blend_type', type=str, 
                        help='the mosaics blending type',
                        default=None)
    parser.add_argument('--output_type', type=str, 
                        help='The output type format',
                        default='jpg')
    parser.add_argument('--output_pattern', type=str, 
                        help='The output file name pattern where "%row%col" will be replaced by "_tr[row]-tc[rol]_" with the row and column numbers',
                        default=None)

    args = parser.parse_args()


    utils.create_dir(args.output_dir)

    render_tiles_2d(args.tiles_file, args.output_dir, args.tile_size, 
        args.output_type, args.jar_file, 
        args.output_pattern, args.blend_type, args.threads_num)

if __name__ == '__main__':
    main()

