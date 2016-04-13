#import mahotas
import glob
import csv
import sys
import os
import json
from decimal import *
from utils import create_dir
import scandir
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

def read_region_metadata_csv_file(fname):
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONE)
            reader.next() # Skip the sep=';' row
            reader.next() # Skip the headers row
            for row in reader:
                yield row


def parse_coordinates_file(input_file, csv_file, section_folder):
    # Find relevant mfovs
    csv_reader = read_region_metadata_csv_file(csv_file)
    max_mfov_num = 0

    relevant_mfovs = []
    for line in csv_reader:
        mfov_num = int(line[0])
        if mfov_num <= 0:
            print("Skipping mfov {} in folder {}".format(mfov_num, section_folder))
            continue

        # Read the mfov number
        max_mfov_num = max(max_mfov_num, mfov_num)
        mfov_folder = os.path.join(section_folder, str(mfov_num).zfill(6))
        if not os.path.exists(mfov_folder):
            print("Error: mfov folder {} not found".format(mfov_folder))
        #elif not verify_mfov_folder(mfov_folder):
        #    print("Error: # of images in mfov directory {} is not 61".format(mfov_folder))
        else:
            relevant_mfovs.append(mfov_num)

    # Read the relevant mfovs tiles locations
    images_dict = {}
    images = []
    x = []
    y = []
    with open(input_file, 'r') as csvfile:
        data_reader = csv.reader(csvfile, delimiter='\t')
        for row in data_reader:
            img_fname = row[0]
            # Make sure that the mfov appears in the relevant mfovs
            if not (img_fname.split('\\')[0]).isdigit():
                # skip the row
                continue
            if int(img_fname.split('\\')[0]) not in relevant_mfovs:
                # skip the row
                continue

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

# Helper functions to parse image folders


def read_image_files(folder):
    # Yields non-thumbnail image files from the given folder
    for entry in scandir.scandir(folder):
        if entry.name.endswith('.bmp') and entry.is_file() and not entry.name.startswith('thumbnail'):
            yield entry.name

def verify_mfov_folder(folder):
    # Make sure that there are 61 (non-zero) file images for the mfov
    img_counter = 0
    for img in read_image_files(folder):
        img_counter += 1
    return img_counter == 61

def read_region_metadata_csv_file(fname):
    #fname = '/n/lichtmanfs2/SCS_2015-9-21_C1_W04_mSEM/_20150930_21-50-13/090_S90R1/region_metadata.csv'
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONE)
            reader.next() # Skip the sep=';' row
            reader.next() # Skip the headers row
            for row in reader:
                # print row
                yield row

def verify_mfovs(folder):
    csv_reader = read_region_metadata_csv_file(os.path.join(folder, "region_metadata.csv"))
    max_mfov_num = 0
    mfovs = []
    for line in csv_reader:
        mfov_num = int(line[0])
        if mfov_num <= 0:
            # print("Skipping mfov {} in folder {}".format(mfov_num, folder))
            continue

        # Read the mfov number
        max_mfov_num = max(max_mfov_num, mfov_num)
        mfov_folder = os.path.join(folder, str(mfov_num).zfill(6))
        if not os.path.exists(mfov_folder):
            print("Error: mfov folder {} not found".format(mfov_folder))
        #elif not verify_mfov_folder(mfov_folder):
        #    print("Error: # of images in mfov directory {} is not 61".format(mfov_folder))
        else:
            mfovs.append(mfov_num)

    return mfovs, max_mfov_num

def find_recent_sections(wafer_dir):
    all_sections = {}
    # The directories are sorted by the timestamp in the directory name. Need to store it in a hashtable for sorting
    all_parent_files = glob.glob(os.path.join(wafer_dir, '*'))
    all_parent_dirs = []
    dir_to_time = {}
    for folder in all_parent_files:
        # Assuming folder names similar to: scs_20151217_19-45-07   (scs can be changed to any other name)
        if os.path.isdir(folder):
            m = re.match('.*_([0-9]{8})_([0-9]{2})-([0-9]{2})-([0-9]{2})$', folder)
            if m is not None:
                dir_to_time[folder] = "{}_{}-{}-{}".format(m.group(1), m.group(2), m.group(3), m.group(4))
                all_parent_dirs.append(folder)
    for sub_folder in sorted(all_parent_dirs, key=lambda folder: dir_to_time[folder]):
        if os.path.isdir(sub_folder):
            print("Finding recent sections from subfolder: {}".format(sub_folder))
            # Get all section folders in the sub-folder
            all_sections_folders = sorted(glob.glob(os.path.join(sub_folder, '*_*')))
            for section_folder in all_sections_folders:
                if os.path.isdir(section_folder):
                    # Found a section directory, now need to find out if it has a focus issue or not
                    # (if it has any sub-dir that is all numbers, it hasn't got an issue)
                    section_num = os.path.basename(section_folder).split('_')[0]
                    relevant_mfovs, max_mfov_num = verify_mfovs(section_folder)
                    if len(relevant_mfovs) > 0:
                        # a good section
                        if min(relevant_mfovs) == 1 and max_mfov_num == len(relevant_mfovs):
                            # The directories in the wafer directory are sorted by the timestamp, and so here we'll get the most recent scan of the section
                            all_sections[section_num] = section_folder
                        else:
                            print("Error while parsing section {} in {}, skipping.".format(section_num, section_folder))
                    #    else:
                    #        missing_mfovs = []
                    #        for i in range(0, max(relevant_mfovs)):
                    #            if i+1 not in relevant_mfovs:
                    #                missing_mfovs.append(str(i+1))
                    #        all_sections[section_num] = MFOVS_MISSING_STR + ':"{}"'.format(','.join(missing_mfovs))
                    #else:
                    #    all_sections[section_num] = FOCUS_FAIL_STR
    return all_sections

def parse_section_folder(wafer_name, section_num_str, layer, section_folder, output_folder):
    csv_file = os.path.join(section_folder, "region_metadata.csv")
    coords_file = os.path.join(section_folder, "full_image_coordinates.txt")
    if os.path.exists(coords_file) and os.path.exists(csv_file):
        output_json_fname = os.path.join(output_folder, "{0}_Sec{1}.json".format(wafer_name, section_num_str))

        if os.path.exists(output_json_fname):
            print "Output file {} already found, skipping".format(output_json_fname)
            return


        images, x, y = parse_coordinates_file(coords_file, csv_file, section_folder)
        # Reset top left to 0,0
        x = offset_list(x)
        y = offset_list(y)
        cur_layer = parse_layer(section_folder, images, x, y)
        #max_layer_width = max(max_layer_width, cur_layer["width"])
        #max_layer_height = max(max_layer_height, cur_layer["height"])
#       all_layers.append(cur_layer)



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
                "layer" : layer,
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
            print('Imported multibeam data to {0}.'.format(output_json_fname))
        else:
            print('Nothing to import ({}).'.format(section_folder))
    else:
        print("Could not find full_image_coordinates.txt, or region_metadata.csv, skipping subfolder: {}".format(section_folder))




def parse_wafer(wafer_folder, output_folder, wafer_name, start_layer=1):
    all_sections = find_recent_sections(wafer_folder)


    # Insert a missing section for each non-seen section number starting from 001 (and ending with the highest number)
    all_sections_keys = sorted(all_sections.keys())
    max_section = int(all_sections_keys[-1])
    print("Max section: {}".format(max_section))
    missing_sections = []
    focus_failed_sections = []
    count = 0
    for i in range(1, max_section + 1):
        section_num_str = str(i).zfill(3)
        if section_num_str in all_sections:
            # Found a section
            print("Parsing section {} from: {}".format(section_num_str, all_sections[section_num_str]))
            parse_section_folder(wafer_name, section_num_str, start_layer + i - 1, all_sections[section_num_str], output_folder)
            count += 1
        else:
            # Found a missing section
            print("Skipping missing section: {}".format(section_num_str))

    print("Parsed {} sections: ".format(count))





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
    wafer_name = 'W' + str(int(m.group(1))).zfill(2)

    print "wafer_name", wafer_name, "start_layer", start_layer
    parse_wafer(input_folder, output_folder, wafer_name, start_layer)


if __name__ == '__main__':
    main()


