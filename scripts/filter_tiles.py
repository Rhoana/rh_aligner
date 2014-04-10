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
        tile_bbox = BoundingBox(tile['boundingBox'])
        if bbox.overlap(tile_bbox):
            relevant_tiles.append(tile)
    return relevant_tiles


def create_single_tile_specs(relevant_tiles, working_dir):
    tile_files = []
    for tile in relevant_tiles:
        out_spec_file = '{0}_tilespec.json'.format(tile['mipmapLevels']['0']['imageUrl'].split(os.path.sep)[-1])
        out_spec_file = os.path.join(working_dir, out_spec_file)
#        print out_spec_file
        with open(out_spec_file, 'w') as outfile:
            tile_to_json = [ tile ]
            json.dump(tile_to_json, outfile)
        tile_files.append(out_spec_file)
    return tile_files



def filter_tiles(tiles_fname, working_dir, bbox):
    # parse the bounding box arguments
    bbox = BoundingBox(bbox)

    # create a workspace directory if not found
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    # load all tiles from the tile-spec json file that are relevant to our bounding box
    relevant_tiles = load_tiles(tiles_fname, bbox)

    # Create a tile-spec file for each relevant tile
    create_single_tile_specs(relevant_tiles, working_dir)

def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Takes a json file that contains many tiles with their bounding boxes (Tile-Spec format)\
        and a bounding box, and outputs a json file for each tile that is overlapping with the bounding box')
    parser.add_argument('tiles_fname', metavar='tiles_json', type=str, 
                        help='a tile_spec file that contains all the images to be aligned in json format')
    parser.add_argument('-w', '--workspace_dir', type=str, 
                        help='a directory where the output files will be kept (default: ./temp)',
                        default='./temp')
    # the default bounding box is as big as the image can be
    parser.add_argument('-b', '--bounding_box', type=str, 
                        help='the bounding box of the part of image that needs to be aligned format: "from_x to_x from_y to_y" (default: all tiles)',
                        default='{0} {1} {2} {3}'.format((-sys.maxint - 1), sys.maxint, (-sys.maxint - 1), sys.maxint))

    args = parser.parse_args()

    #print args

    filter_tiles(args.tiles_fname, args.workspace_dir, args.bounding_box)

if __name__ == '__main__':
    main()

