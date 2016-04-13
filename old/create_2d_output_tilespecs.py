from __future__ import print_function
import glob
import sys
import os
import json
import re
import tifffile # fast peeking at tiff sizes
import urlparse, urllib
import cv2
import argparse
import utils

def get_image_size(image_file):
    img = cv2.imread(image_file, 0)
    return img.shape


def path2url(path):
    return urlparse.urljoin('file:', urllib.pathname2url(path))

def extract_coords(filename, image_size):
    m = re.match('.*_tr([0-9]+)-tc([0-9]+)_.*[.].*', os.path.basename(filename))
    offset_y = (int(m.group(1)) - 1) * image_size[0]
    offset_x = (int(m.group(2)) - 1) * image_size[1]
    return int(offset_x), int(offset_y)

def find_image_files(subdir):
    all_files = []
    all_files.extend(glob.glob(os.path.join(subdir, '*.jpg')))
    all_files.extend(glob.glob(os.path.join(subdir, '*.tif')))
    all_files.extend(glob.glob(os.path.join(subdir, '*.png')))
    return all_files

def write_tilespec_from_folder(subdir, output_json_fname, layer):
    '''Writes the tilespec for a single directory (aka, section)'''
    tilespecs = []
    image_size = None

    if os.path.exists(output_json_fname):
        print("Will not overwrite {}".format(output_json_fname))
        return

    for image_file in sorted(find_image_files(subdir)):
        if image_size is None:
            image_size = get_image_size(image_file)
        coords = extract_coords(image_file, image_size)
        tilespec = {
            "mipmapLevels" : {
                "0" : {
                    "imageUrl" : path2url(os.path.abspath(image_file)),
                    }
                },
            "minIntensity" : 0.0,
            "maxIntensity" : 255.0,
            "transforms" : [{
                    "className" : "mpicbg.trakem2.transform.TranslationModel2D",
                    # x, y offset of upper right corner
                    "dataString" : "{0} {1}".format(float(coords[0]), float(coords[1]))
                    }],
            "layer" : layer,
            "width" : image_size[1],
            "height" : image_size[0],
            # BoundingBox in the format "from_x to_x from_y to_y" (left right top bottom)
            "bbox" : [coords[0], coords[0] + image_size[1],
                      coords[1], coords[1] + image_size[0]]
            }
        tilespecs.append(tilespec)

    if len(tilespecs) > 0:
        with open(output_json_fname, 'w') as outjson:
            json.dump(tilespecs, outjson, sort_keys=True, indent=4)
            print('Wrote tilespec to {0}'.format(output_json_fname))
    else:
        print('Nothing to write in directory {}'.format(subdir))


def write_tilespec_from_file(image_file, output_json_fname, layer):
    '''Writes the tilespec for a single image/section'''
    tilespecs = []
    image_size = None

    if os.path.exists(output_json_fname):
        print("Will not overwrite {}".format(output_json_fname))
        return

    image_size = get_image_size(image_file)

    tilespec = {
        "mipmapLevels" : {
            "0" : {
                "imageUrl" : path2url(os.path.abspath(image_file)),
                }
            },
        "minIntensity" : 0.0,
        "maxIntensity" : 255.0,
        "transforms" : [{
                "className" : "mpicbg.trakem2.transform.TranslationModel2D",
                # x, y offset of upper right corner
                "dataString" : "0 0"
                }],
        "layer" : layer,
        "width" : image_size[1],
        "height" : image_size[0],
        # BoundingBox in the format "from_x to_x from_y to_y" (left right top bottom)
        "bbox" : [0, image_size[1],
                  0, image_size[0]]
        }
    tilespecs.append(tilespec)

    with open(output_json_fname, 'w') as outjson:
        json.dump(tilespecs, outjson, sort_keys=True, indent=4)
        print('Wrote tilespec to {0}'.format(output_json_fname))





def create_2d_output_tilespecs(json_input_dir, render_input_dir, output_dir):
    all_files_folders = glob.glob(os.path.join(render_input_dir, '*'))
    all_files_folders.sort(key=lambda s: s.lower())
    for file_or_folder in all_files_folders:
        json_path = os.path.join(json_input_dir, os.path.basename(file_or_folder) + '.json')
        if not (os.path.exists(json_path) and os.path.isfile(json_path)):
            print('Error: could not find the json file for {} (the search for {} has failed).'.format(file_or_folder, json_path))

        layer = utils.read_layer_from_file(json_path)

        output_path = os.path.join(output_dir, os.path.basename(file_or_folder) + '.json')
        if os.path.isdir(file_or_folder):
            write_tilespec_from_folder(os.path.join(file_or_folder, '0'), output_path, layer)
        elif os.path.isfile(file_or_folder):
            write_tilespec_from_file(file_or_folder, output_path, layer)

def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Create tilespecs to tiles that were rendered after the 2d alignment process.\
                        Will be used for 3d alignment. Assumes no-overlap, and no transformation between the adjacent tiles.')
    parser.add_argument('json_input_dir', metavar='json_input_dir', type=str,
                        help='The 2d alignment process json output directory (each json file contains the tiles after alignment)')
    parser.add_argument('render_input_dir', metavar='render_input_dir', type=str,
                        help='The 2d alignment process rendered images output directory (either contains the rendered sections, or folders with the rendered tiles in them)')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='The directory where the json will be stored (default: ./output_json)',
                        default='./output_json')

    args = parser.parse_args()

    utils.create_dir(args.output_dir)

    create_2d_output_tilespecs(args.json_input_dir, args.render_input_dir, args.output_dir)

if __name__ == '__main__':
    main()
