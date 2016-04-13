# Takes a json file that contains many tiles with their bounding boxes (Tile-Spec format)
# and a bounding box, and outputs a json file for each tile that is overlapping with the bounding box


import sys
import os
import argparse
import json
from bounding_box import BoundingBox


# common functions


def load_tiles(tiles_spec_fname, bbox):
    relevant_tiles = []
    with open(tiles_spec_fname, 'r') as data_file:
        data = json.load(data_file)
    for tile in data:
        tile_bbox = BoundingBox.fromList(tile['bbox'])
        if bbox.overlap(tile_bbox):
            relevant_tiles.append(tile)
    return relevant_tiles


def filter_tiles(tiles_fname, out_fname, bbox):
    # parse the bounding box arguments
    bbox = BoundingBox.fromStr(bbox)

    # load all tiles from the tile-spec json file that are relevant to our bounding box
    relevant_tiles = load_tiles(tiles_fname, bbox)

    # Create a tile-spec file that includes all relevant tiles
    with open(out_fname, 'w') as outfile:
        json.dump(relevant_tiles, outfile, sort_keys=True, indent=4)

def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Takes a json file that contains many tiles with their bounding boxes (Tile-Spec format)\
        and a bounding box, and outputs a json file for each tile that is overlapping with the bounding box')
    parser.add_argument('tiles_fname', metavar='tiles_json', type=str, 
                        help='a tile_spec file that contains all the images to be aligned in json format')
    parser.add_argument('-o', '--output_file', type=str, 
                        help='an output tile_spec file, that will include only the relevant tiles (default: ./filtered.json)',
                        default='./filtered.json')
    # the default bounding box is as big as the image can be
    parser.add_argument('-b', '--bounding_box', type=str, 
                        help='the bounding box of the part of image that needs to be aligned format: "from_x to_x from_y to_y" (default: all tiles)',
                        default='{0} {1} {2} {3}'.format((-sys.maxint - 1), sys.maxint, (-sys.maxint - 1), sys.maxint))

    args = parser.parse_args()

    #print args

    filter_tiles(args.tiles_fname, args.output_file, args.bounding_box)

if __name__ == '__main__':
    main()

