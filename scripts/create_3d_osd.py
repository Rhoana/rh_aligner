# Receives a directory which contains directoris of tiled images in different zoom levels (each subdir has zoom levels starting from 0,
# of some section), and creates the open_sea_dragon (osd) tilesource file that fits these sections
#
# Note that all tiles in a section are assumed to be of the same size (and tile width == tile height),
# and that the given directory includes a sub-directory named "0" that has the full resultion tiles
# 

import sys
import os
import glob
import argparse
import utils
import math
import numpy as np
import cv2


def get_tile_size(tile_file):
    img = cv2.imread(tile_file, 0)
    return img.shape[0]


def write_open_sea_dragon_section_func(out_data, section_count, dir_name, tile_size, initial_rows, initial_cols, zoom_level):
    out_data.write('function createTileSource_{0}() {{\n'.format(section_count))
    out_data.write('   var tileSource = {};\n')
    out_data.write('   tileSource.height = {}\n'.format(initial_rows * tile_size))
    out_data.write('   tileSource.width = {}\n'.format(initial_cols * tile_size))
    out_data.write('   tileSource.tileSize = {}\n'.format(tile_size))
    out_data.write('   tileSource.tileOverlap = 0\n')
    out_data.write('   tileSource.minLevel = 0\n')
    out_data.write('   tileSource.maxLevel = {}\n'.format(zoom_level))
    out_data.write('   tileSource.getTileUrl = function( level, x, y ) {\n')
    out_data.write('       return "/tiles3d/{0}/" + ({1} - level) + "/tile_" + y + "_" + x + ".jpg";\n'.format(dir_name, zoom_level))
    out_data.write('   }\n')
    out_data.write('   return tileSource;\n')
    out_data.write('}\n\n')



def create_3d_osd(sections_dir):
    all_dirs = [ d for d in glob.glob(os.path.join(sections_dir, '*')) if os.path.isdir(d) ]

    all_dirs.sort()

    # Create the output file
    out_file = os.path.join(sections_dir, "osd.js")
    out_data = open(out_file, 'w')
    #out_data.write('\n')


    section_count = 0

    for d in all_dirs:
        print "Parsing directory: {}".format(d)
        first_tile = glob.glob(os.path.join(os.path.join(d, '0'), 'tile_0_0.*'))[0]
        tile_size = get_tile_size(first_tile)

        # get number of rows and columns used for full resolution
        initial_rows = len(glob.glob(os.path.join(os.path.join(d, '0'), 'tile_*_0.*')))
        initial_cols = len(glob.glob(os.path.join(os.path.join(d, '0'), 'tile_0_*')))

        # get number of zoom levels
        zoom_levels = len([ folder for folder in glob.glob(os.path.join(d, '*')) if os.path.isdir(folder) ]) - 1

        dir_name = d[len(sections_dir):]

        write_open_sea_dragon_section_func(out_data, section_count, dir_name, tile_size, initial_rows, initial_cols, zoom_levels)
        section_count += 1


    # write the function to return an array of all tile sources
    out_data.write('function createTileSource() {\n')
    out_data.write('   var tileSources = [];\n')
    for i in range(section_count):
        out_data.write('   tileSources.push(createTileSource_{0}());\n'.format(i))
    out_data.write('   return tileSources;\n')
    out_data.write('}\n')

    out_data.close()



def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Receives a directory which contains sub directories with tiled sections, \
                        and creates a single 3d osd.js file for open_sea_dragon tileSources.')
    parser.add_argument('sections_dir', metavar='sections_dir', type=str, 
                        help='the directory which has a sub-directory named "0" that contains the full resolution tiles')


    args = parser.parse_args()

    #print args

    create_3d_osd(args.sections_dir)

if __name__ == '__main__':
    main()

