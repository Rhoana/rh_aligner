#import mahotas
import glob
import csv
import sys
import os
import json
from decimal import *
from utils import create_dir
import re

# input_folder: the folder of a single wafer
# output_folder: where to put the tilespecs


def read_bmp_dimensions(bmp_file):
    """ Taken from: http://runnable.com/UqJdRnCIohYmAAGP/reading-binary-files-in-python-for-io """
    import struct
    dims = None
    # When reading a binary file, always add a 'b' to the file open mode
    with open(bmp_file, 'rb') as f:
        # BMP files store their width and height statring at byte 18 (12h), so seek
        # to that position
        f.seek(18)

        # The width and height are 4 bytes each, so read 8 bytes to get both of them
        bytes = f.read(8)

        # Here, we decode the byte array from the last step. The width and height
        # are each unsigned, little endian, 4 byte integers, so they have the format
        # code '<II'. See http://docs.python.org/3/library/struct.html for more info
        # Our test showed that we needed to use 2's complement (so using 'i' instead of 'I', and taking absolute value)
        size = struct.unpack('<ii', bytes)
        size = [abs(size[0]), abs(size[1])]

        dims = [size[1], size[0]]
    return dims

def read_dimensions(img_file):
    if img_file.lower().endswith(".bmp"):
        return read_bmp_dimensions(img_file)
    else:
        import cv2
        im = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        if im is None:
            print('Cannot read tile image: {}'.format(image_file))
            sys.exit(1)
        return im.shape


def filename_decimal_key(tile):
    fname = tile["file_base_name"]
    return Decimal(''.join([c for c in fname if c.isdigit()]))

def parse_layer(base_dir, images, x, y):
    '''Writes the tilespec for a single section'''
    tiles = []
    layer_size = [0, 0]
    image_size = None

    for i, img in enumerate(images):
        # If not using windows, change the folder separator
        if os.name == "posix":
            img = img.replace('\\', '/')

        image_file = os.path.join(base_dir, img)
        tile = {}
        if image_size is None:
            image_size = read_dimensions(image_file)
        tile["file_full_path"] = os.path.abspath(image_file)
        tile["file_base_name"] = os.path.basename(tile["file_full_path"])
        tile["width"] = image_size[1]
        tile["height"] = image_size[0]
        tile["tx"] = x[i]
        tile["ty"] = y[i]
        tile["mfov"] = int(tile["file_base_name"].split('_')[1])
        tile["tile_index"] = int(tile["file_base_name"].split('_')[2])
        tiles.append(tile)
        layer_size[0] = max(layer_size[0], image_size[0] + tile["ty"])
        layer_size[1] = max(layer_size[1], image_size[1] + tile["tx"])

    # if len(tiles) > 0:
    #     write_layer(out_data, layer, tiles)
    # else:
    #     print('Nothing to write from directory {}'.format(subdir))
    if len(tiles) == 0:
        print('Nothing to write from directory {}'.format(base_dir))
    
    tiles.sort(key=filename_decimal_key)    
    layer_data = {}
    layer_data["height"] = layer_size[0]
    layer_data["width"] = layer_size[1]
    layer_data["tiles"] = tiles
    return layer_data


def parse_coordinates_file(input_file):
    images_dict = {}
    images = []
    x = []
    y = []
    with open(input_file, 'r') as csvfile:
        data_reader = csv.reader(csvfile, delimiter='\t')
        for row in data_reader:
            img_fname = row[0]
            img_sec_mfov_beam = '_'.join(img_fname.split('\\')[-1].split('_')[:3])
            # Make sure that no duplicates appear
            if img_sec_mfov_beam not in images_dict.keys():
                images.append(img_fname)
                images_dict[img_sec_mfov_beam] = img_fname
                cur_x = float(row[1])
                cur_y = float(row[2])
                x.append(cur_x)
                y.append(cur_y)
            else:
                # Either the image is duplicated, or a newer version was taken,
                # so make sure that the newer version is used
                prev_img = images_dict[img_sec_mfov_beam]
                prev_img_date = prev_img.split('\\')[-1].split('_')[-1]
                curr_img_date = img_fname.split('\\')[-1].split('_')[-1]
                if curr_img_date > prev_img_date:
                    idx = images.index(prev_img)
                    images[idx] = img_fname
                    images_dict[img_sec_mfov_beam] = img_fname

    return images, x, y

def offset_list(lst):
    m = min(lst)
    return [item - m for item in lst]


def parse_wafer(wafer_folder, output_folder, wafer_name=1, start_layer=1):
    sub_folders = sorted(glob.glob(os.path.join(wafer_folder, '*')))

    coords = {}

    min_x = None
    min_y = None

    tile_depth = 1
    max_section = start_layer


#    all_layers = []

    for sub_folder in sub_folders:
        if os.path.isdir(sub_folder):
            print("Parsing subfolder: {}".format(sub_folder))
            coords_file = os.path.join(sub_folder, "full_image_coordinates.txt")
            if os.path.exists(coords_file):
                section_dir_name = sub_folder.split(os.path.sep)[-1]
                m = re.match('([0-9]+)_S([0-9]+)R.*', section_dir_name)
                layer = int(m.group(2))
                output_json_fname = os.path.join(output_folder, "{0}_Sec{1:03d}.json".format(wafer_name, layer))

                if os.path.exists(output_json_fname):
                    print "Output file {} already found, skipping".format(output_json_fname)
                    continue

                images, x, y = parse_coordinates_file(coords_file)
                # Reset top left to 0,0
                x = offset_list(x)
                y = offset_list(y)
                cur_layer = parse_layer(sub_folder, images, x, y)
                #max_layer_width = max(max_layer_width, cur_layer["width"])
                #max_layer_height = max(max_layer_height, cur_layer["height"])
                cur_layer["layer_num"] = layer + start_layer - 1
#                all_layers.append(cur_layer)



                export = []

                for tile in cur_layer["tiles"]:
                    tilespec = {
                        "mipmapLevels" : {
                            "0" : {
                                "imageUrl" : "file://{0}".format(tile["file_full_path"].replace(os.path.sep, '/'))
                                #"maskUrl" : "file://",
                            }
                        },
                        "minIntensity" : 0.0,
                        "maxIntensity" : 255.0,
                        "layer" : cur_layer["layer_num"],
                        "transforms" : [{
                            "className" : "mpicbg.trakem2.transform.TranslationModel2D",
                            "dataString" : "{0} {1}".format(tile["tx"], tile["ty"])
                        }],
                        "width" : tile["width"],
                        "height" : tile["height"],
                        "mfov" : tile["mfov"],
                        "tile_index" : tile["tile_index"],
                        # BoundingBox in the format "from_x to_x from_y to_y" (left right top bottom)
                        "bbox" : [ tile["tx"], tile["tx"] + tile["width"],
                            tile["ty"], tile["ty"] + tile["height"] ]
                    }

                    export.append(tilespec)

                if len(export) > 0:
                    with open(output_json_fname, 'w') as outjson:
                        json.dump(export, outjson, sort_keys=True, indent=4)
                    'Imported multibeam data to {0}.'.format(output_json_fname)
                else:
                    print 'Nothing to import.'
            else:
                print "Could not find full_image_coordinates.txt, skipping subfolder: {}".format(sub_folder)




def main():
    input_folder = sys.argv[1]

    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    else:
        output_folder = os.path.join(input_folder, 'tilespecs')

    if len(sys.argv) > 3:
        start_layer = int(sys.argv[-1])
    else:
        start_layer = 1

    create_dir(output_folder)

    print input_folder
    m = re.match('.*[W|w]([0-9]+).*', input_folder)
    if m is None:
        print "Could not find a wafer number, assuming it is in wafer 01"
        wafer_name = 'W01'
    else:
        wafer_name = 'W' + str(int(m.group(1))).zfill(2)

    print "wafer_name", wafer_name, "start_layer", start_layer
    parse_wafer(input_folder, output_folder, wafer_name, start_layer)

if __name__ == '__main__':
    main()


