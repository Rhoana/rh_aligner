#import mahotas
import glob
import csv
import sys
import os
import json
from decimal import *


input_folder = sys.argv[1]
output_json_fname = sys.argv[-1]

if len(sys.argv) < 3:
    output_json_fname = os.path.join(input_folder, 'tilespec.json')
else:
    output_json_fname = sys.argv[-1]

scan_index = 0

if len(sys.argv) > 3:
    scan_index = int(sys.argv[2])

sub_folders = sorted(glob.glob(os.path.join(input_folder, '*')))

coords = {}

min_x = None
min_y = None

tile_depth = 1

for sub_folder in sub_folders:
    if os.path.isdir(sub_folder):
        metadata_file = os.path.join(sub_folder, 'metadata.txt')
        pixel_coordinates_file = os.path.join(sub_folder, 'pixelCoordinates.txt')

        if not os.path.exists(metadata_file) or not os.path.exists(pixel_coordinates_file):
            continue

        # Read metadata
        pixel_size = 4.

        with open(metadata_file) as infile:
            metadata_reader = csv.reader(infile, delimiter='\t')
            for row in metadata_reader:
                if row[0] == 'Pixelsize:':
                    pixel_size = Decimal(row[-1].replace('nm',''))
                if row[0] == 'Width:':
                    tile_width = int(row[-1].replace('px',''))
                if row[0] == 'Height:':
                    tile_height = int(row[-1].replace('px',''))

        with open(pixel_coordinates_file) as infile:
            pix_coord_reader = csv.reader(infile, delimiter='\t')
            for row in pix_coord_reader:
                tile_fname = row[0]
                tile_full_path = os.path.join(sub_folder, tile_fname)
                if os.path.exists(tile_full_path):
                    scan_number = int(Decimal(row[3]))
                    raw_coords = [Decimal(row[1]), Decimal(row[2]), Decimal(row[3])]
                    coords[tile_full_path] = raw_coords

                    if min_x is None or raw_coords[0] < min_x:
                        min_x = raw_coords[0]
                    if min_y is None or raw_coords[1] < min_y:
                        min_y = raw_coords[1]

        print 'Read hex data from {0}.'.format(sub_folder)

export = []

def filename_decimal_key(path):
    fname = os.path.split(path)[-1]
    return Decimal(''.join([c for c in fname if c.isdigit()]))

# Reset top left to 0,0
print 'Offsetting by ({0}, {1}).'.format(min_x, min_y)
tile_order = coords.keys()
tile_order.sort(key=filename_decimal_key)
for tile_full_path in tile_order:
    raw_coords = coords[tile_full_path]

    #Offset
    raw_coords[0] -= min_x
    raw_coords[1] -= min_y

    tilespec = {
        "mipmapLevels" : {
            "0" : {
                "imageUrl" : "file://{0}".format(tile_full_path.replace(os.path.sep, '/'))
                #"maskUrl" : "file://",
            }
        },
        "minIntensity" : 0.0,
        "maxIntensity" : 255.0,
        "transforms" : [{
            "className" : "mpicbg.trakem2.transform.TranslationModel2D",
            "dataString" : "{0} {1}".format(raw_coords[0], raw_coords[1])
        }],
        # BoundingBox in the format "from_x to_x from_y to_y" (left right top bottom)
        "bbox" : [raw_coords[0], raw_coords[0] + tile_width,
            raw_coords[1], raw_coords[1] + tile_height]
    }

    export.append(tilespec)

if len(export) > 0:
    with open(output_json_fname, 'w') as outjson:
        json.dump(export, outjson, sort_keys=True, indent=4)
    'Imported multibeam data to {0}.'.format(output_json_fname)
else:
    print 'Nothing to import.'


