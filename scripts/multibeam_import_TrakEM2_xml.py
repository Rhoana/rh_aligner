from __future__ import print_function
import glob
import sys
import os
import re
from decimal import *
import urlparse, urllib
import argparse
import csv
import math
import time

# a global object id counter for the different objects for TrakEM2
global_oid_counter = 5

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


def fetch_and_increase_oid():
    global global_oid_counter
    res = global_oid_counter
    global_oid_counter += 1
    return res

def write_pre_xml(out_data, xml_file_name, layer_width, layer_height):
    unuid="{}.1224689168.2106301".format(int(round(time.time() * 1000)))
    lines = """<?xml version="1.0" encoding="ISO-8859-1"?>
<!DOCTYPE trakem2_anything [
        <!ELEMENT trakem2 (project,t2_layer_set,t2_display)>
        <!ELEMENT project (anything)>
        <!ATTLIST project id NMTOKEN #REQUIRED>
        <!ATTLIST project unuid NMTOKEN #REQUIRED>
        <!ATTLIST project title NMTOKEN #REQUIRED>
        <!ATTLIST project preprocessor NMTOKEN #REQUIRED>
        <!ATTLIST project mipmaps_folder NMTOKEN #REQUIRED>
        <!ATTLIST project storage_folder NMTOKEN #REQUIRED>
        <!ATTLIST project n_mipmap_threads NMTOKEN #REQUIRED>
        <!ATTLIST project look_ahead_cache NMTOKEN #REQUIRED>
        <!ATTLIST project n_undo_steps NMTOKEN #REQUIRED>
        <!ELEMENT anything EMPTY>
        <!ATTLIST anything id NMTOKEN #REQUIRED>
        <!ATTLIST anything expanded NMTOKEN #REQUIRED>
        <!ELEMENT t2_layer (t2_patch,t2_label,t2_layer_set,t2_profile)>
        <!ATTLIST t2_layer oid NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer thickness NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer z NMTOKEN #REQUIRED>
        <!ELEMENT t2_layer_set (t2_prop,t2_linked_prop,t2_annot,t2_layer,t2_pipe,t2_ball,t2_area_list,t2_calibration,t2_stack,t2_treeline)>
        <!ATTLIST t2_layer_set oid NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set layer_id NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set transform NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set style NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set locked NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set visible NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set title NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set links NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set composite NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set layer_width NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set layer_height NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set rot_x NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set rot_y NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set rot_z NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set snapshots_quality NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set color_cues NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set area_color_cues NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set avoid_color_cue_colors NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set n_layers_color_cue NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set paint_arrows NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set paint_tags NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set paint_edge_confidence_boxes NMTOKEN #REQUIRED>
        <!ATTLIST t2_layer_set preload_ahead NMTOKEN #REQUIRED>
        <!ELEMENT t2_calibration EMPTY>
        <!ATTLIST t2_calibration pixelWidth NMTOKEN #REQUIRED>
        <!ATTLIST t2_calibration pixelHeight NMTOKEN #REQUIRED>
        <!ATTLIST t2_calibration pixelDepth NMTOKEN #REQUIRED>
        <!ATTLIST t2_calibration xOrigin NMTOKEN #REQUIRED>
        <!ATTLIST t2_calibration yOrigin NMTOKEN #REQUIRED>
        <!ATTLIST t2_calibration zOrigin NMTOKEN #REQUIRED>
        <!ATTLIST t2_calibration info NMTOKEN #REQUIRED>
        <!ATTLIST t2_calibration valueUnit NMTOKEN #REQUIRED>
        <!ATTLIST t2_calibration timeUnit NMTOKEN #REQUIRED>
        <!ATTLIST t2_calibration unit NMTOKEN #REQUIRED>
        <!ELEMENT t2_ball (t2_prop,t2_linked_prop,t2_annot,t2_ball_ob)>
        <!ATTLIST t2_ball oid NMTOKEN #REQUIRED>
        <!ATTLIST t2_ball layer_id NMTOKEN #REQUIRED>
        <!ATTLIST t2_ball transform NMTOKEN #REQUIRED>
        <!ATTLIST t2_ball style NMTOKEN #REQUIRED>
        <!ATTLIST t2_ball locked NMTOKEN #REQUIRED>
        <!ATTLIST t2_ball visible NMTOKEN #REQUIRED>
        <!ATTLIST t2_ball title NMTOKEN #REQUIRED>
        <!ATTLIST t2_ball links NMTOKEN #REQUIRED>
        <!ATTLIST t2_ball composite NMTOKEN #REQUIRED>
        <!ATTLIST t2_ball fill NMTOKEN #REQUIRED>
        <!ELEMENT t2_ball_ob EMPTY>
        <!ATTLIST t2_ball_ob x NMTOKEN #REQUIRED>
        <!ATTLIST t2_ball_ob y NMTOKEN #REQUIRED>
        <!ATTLIST t2_ball_ob r NMTOKEN #REQUIRED>
        <!ATTLIST t2_ball_ob layer_id NMTOKEN #REQUIRED>
        <!ELEMENT t2_label (t2_prop,t2_linked_prop,t2_annot)>
        <!ATTLIST t2_label oid NMTOKEN #REQUIRED>
        <!ATTLIST t2_label layer_id NMTOKEN #REQUIRED>
        <!ATTLIST t2_label transform NMTOKEN #REQUIRED>
        <!ATTLIST t2_label style NMTOKEN #REQUIRED>
        <!ATTLIST t2_label locked NMTOKEN #REQUIRED>
        <!ATTLIST t2_label visible NMTOKEN #REQUIRED>
        <!ATTLIST t2_label title NMTOKEN #REQUIRED>
        <!ATTLIST t2_label links NMTOKEN #REQUIRED>
        <!ATTLIST t2_label composite NMTOKEN #REQUIRED>
        <!ELEMENT t2_filter EMPTY>
        <!ELEMENT t2_patch (t2_prop,t2_linked_prop,t2_annot,ict_transform,ict_transform_list,t2_filter)>
        <!ATTLIST t2_patch oid NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch layer_id NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch transform NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch style NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch locked NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch visible NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch title NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch links NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch composite NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch file_path NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch original_path NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch type NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch false_color NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch ct NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch o_width NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch o_height NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch min NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch max NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch o_width NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch o_height NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch pps NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch mres NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch ct_id NMTOKEN #REQUIRED>
        <!ATTLIST t2_patch alpha_mask_id NMTOKEN #REQUIRED>
        <!ELEMENT t2_pipe (t2_prop,t2_linked_prop,t2_annot)>
        <!ATTLIST t2_pipe oid NMTOKEN #REQUIRED>
        <!ATTLIST t2_pipe layer_id NMTOKEN #REQUIRED>
        <!ATTLIST t2_pipe transform NMTOKEN #REQUIRED>
        <!ATTLIST t2_pipe style NMTOKEN #REQUIRED>
        <!ATTLIST t2_pipe locked NMTOKEN #REQUIRED>
        <!ATTLIST t2_pipe visible NMTOKEN #REQUIRED>
        <!ATTLIST t2_pipe title NMTOKEN #REQUIRED>
        <!ATTLIST t2_pipe links NMTOKEN #REQUIRED>
        <!ATTLIST t2_pipe composite NMTOKEN #REQUIRED>
        <!ATTLIST t2_pipe d NMTOKEN #REQUIRED>
        <!ATTLIST t2_pipe p_width NMTOKEN #REQUIRED>
        <!ATTLIST t2_pipe layer_ids NMTOKEN #REQUIRED>
        <!ELEMENT t2_polyline (t2_prop,t2_linked_prop,t2_annot)>
        <!ATTLIST t2_polyline oid NMTOKEN #REQUIRED>
        <!ATTLIST t2_polyline layer_id NMTOKEN #REQUIRED>
        <!ATTLIST t2_polyline transform NMTOKEN #REQUIRED>
        <!ATTLIST t2_polyline style NMTOKEN #REQUIRED>
        <!ATTLIST t2_polyline locked NMTOKEN #REQUIRED>
        <!ATTLIST t2_polyline visible NMTOKEN #REQUIRED>
        <!ATTLIST t2_polyline title NMTOKEN #REQUIRED>
        <!ATTLIST t2_polyline links NMTOKEN #REQUIRED>
        <!ATTLIST t2_polyline composite NMTOKEN #REQUIRED>
        <!ATTLIST t2_polyline d NMTOKEN #REQUIRED>
        <!ELEMENT t2_profile (t2_prop,t2_linked_prop,t2_annot)>
        <!ATTLIST t2_profile oid NMTOKEN #REQUIRED>
        <!ATTLIST t2_profile layer_id NMTOKEN #REQUIRED>
        <!ATTLIST t2_profile transform NMTOKEN #REQUIRED>
        <!ATTLIST t2_profile style NMTOKEN #REQUIRED>
        <!ATTLIST t2_profile locked NMTOKEN #REQUIRED>
        <!ATTLIST t2_profile visible NMTOKEN #REQUIRED>
        <!ATTLIST t2_profile title NMTOKEN #REQUIRED>
        <!ATTLIST t2_profile links NMTOKEN #REQUIRED>
        <!ATTLIST t2_profile composite NMTOKEN #REQUIRED>
        <!ATTLIST t2_profile d NMTOKEN #REQUIRED>
        <!ELEMENT t2_area_list (t2_prop,t2_linked_prop,t2_annot,t2_area)>
        <!ATTLIST t2_area_list oid NMTOKEN #REQUIRED>
        <!ATTLIST t2_area_list layer_id NMTOKEN #REQUIRED>
        <!ATTLIST t2_area_list transform NMTOKEN #REQUIRED>
        <!ATTLIST t2_area_list style NMTOKEN #REQUIRED>
        <!ATTLIST t2_area_list locked NMTOKEN #REQUIRED>
        <!ATTLIST t2_area_list visible NMTOKEN #REQUIRED>
        <!ATTLIST t2_area_list title NMTOKEN #REQUIRED>
        <!ATTLIST t2_area_list links NMTOKEN #REQUIRED>
        <!ATTLIST t2_area_list composite NMTOKEN #REQUIRED>
        <!ATTLIST t2_area_list fill_paint NMTOKEN #REQUIRED>
        <!ELEMENT t2_area (t2_path)>
        <!ATTLIST t2_area layer_id NMTOKEN #REQUIRED>
        <!ELEMENT t2_path EMPTY>
        <!ATTLIST t2_path d NMTOKEN #REQUIRED>
        <!ELEMENT t2_dissector (t2_prop,t2_linked_prop,t2_annot,t2_dd_item)>
        <!ATTLIST t2_dissector oid NMTOKEN #REQUIRED>
        <!ATTLIST t2_dissector layer_id NMTOKEN #REQUIRED>
        <!ATTLIST t2_dissector transform NMTOKEN #REQUIRED>
        <!ATTLIST t2_dissector style NMTOKEN #REQUIRED>
        <!ATTLIST t2_dissector locked NMTOKEN #REQUIRED>
        <!ATTLIST t2_dissector visible NMTOKEN #REQUIRED>
        <!ATTLIST t2_dissector title NMTOKEN #REQUIRED>
        <!ATTLIST t2_dissector links NMTOKEN #REQUIRED>
        <!ATTLIST t2_dissector composite NMTOKEN #REQUIRED>
        <!ELEMENT t2_dd_item EMPTY>
        <!ATTLIST t2_dd_item radius NMTOKEN #REQUIRED>
        <!ATTLIST t2_dd_item tag NMTOKEN #REQUIRED>
        <!ATTLIST t2_dd_item points NMTOKEN #REQUIRED>
        <!ELEMENT t2_stack (t2_prop,t2_linked_prop,t2_annot,(iict_transform|iict_transform_list)?)>
        <!ATTLIST t2_stack oid NMTOKEN #REQUIRED>
        <!ATTLIST t2_stack layer_id NMTOKEN #REQUIRED>
        <!ATTLIST t2_stack transform NMTOKEN #REQUIRED>
        <!ATTLIST t2_stack style NMTOKEN #REQUIRED>
        <!ATTLIST t2_stack locked NMTOKEN #REQUIRED>
        <!ATTLIST t2_stack visible NMTOKEN #REQUIRED>
        <!ATTLIST t2_stack title NMTOKEN #REQUIRED>
        <!ATTLIST t2_stack links NMTOKEN #REQUIRED>
        <!ATTLIST t2_stack composite NMTOKEN #REQUIRED>
        <!ATTLIST t2_stack file_path CDATA #REQUIRED>
        <!ATTLIST t2_stack depth CDATA #REQUIRED>
        <!ELEMENT t2_tag EMPTY>
        <!ATTLIST t2_tag name NMTOKEN #REQUIRED>
        <!ATTLIST t2_tag key NMTOKEN #REQUIRED>
        <!ELEMENT t2_node (t2_area*,t2_tag*)>
        <!ATTLIST t2_node x NMTOKEN #REQUIRED>
        <!ATTLIST t2_node y NMTOKEN #REQUIRED>
        <!ATTLIST t2_node lid NMTOKEN #REQUIRED>
        <!ATTLIST t2_node c NMTOKEN #REQUIRED>
        <!ATTLIST t2_node r NMTOKEN #IMPLIED>
        <!ELEMENT t2_treeline (t2_node*,t2_prop,t2_linked_prop,t2_annot)>
        <!ATTLIST t2_treeline oid NMTOKEN #REQUIRED>
        <!ATTLIST t2_treeline layer_id NMTOKEN #REQUIRED>
        <!ATTLIST t2_treeline transform NMTOKEN #REQUIRED>
        <!ATTLIST t2_treeline style NMTOKEN #REQUIRED>
        <!ATTLIST t2_treeline locked NMTOKEN #REQUIRED>
        <!ATTLIST t2_treeline visible NMTOKEN #REQUIRED>
        <!ATTLIST t2_treeline title NMTOKEN #REQUIRED>
        <!ATTLIST t2_treeline links NMTOKEN #REQUIRED>
        <!ATTLIST t2_treeline composite NMTOKEN #REQUIRED>
        <!ELEMENT t2_areatree (t2_node*,t2_prop,t2_linked_prop,t2_annot)>
        <!ATTLIST t2_areatree oid NMTOKEN #REQUIRED>
        <!ATTLIST t2_areatree layer_id NMTOKEN #REQUIRED>
        <!ATTLIST t2_areatree transform NMTOKEN #REQUIRED>
        <!ATTLIST t2_areatree style NMTOKEN #REQUIRED>
        <!ATTLIST t2_areatree locked NMTOKEN #REQUIRED>
        <!ATTLIST t2_areatree visible NMTOKEN #REQUIRED>
        <!ATTLIST t2_areatree title NMTOKEN #REQUIRED>
        <!ATTLIST t2_areatree links NMTOKEN #REQUIRED>
        <!ATTLIST t2_areatree composite NMTOKEN #REQUIRED>
        <!ELEMENT t2_connector (t2_node*,t2_prop,t2_linked_prop,t2_annot)>
        <!ATTLIST t2_connector oid NMTOKEN #REQUIRED>
        <!ATTLIST t2_connector layer_id NMTOKEN #REQUIRED>
        <!ATTLIST t2_connector transform NMTOKEN #REQUIRED>
        <!ATTLIST t2_connector style NMTOKEN #REQUIRED>
        <!ATTLIST t2_connector locked NMTOKEN #REQUIRED>
        <!ATTLIST t2_connector visible NMTOKEN #REQUIRED>
        <!ATTLIST t2_connector title NMTOKEN #REQUIRED>
        <!ATTLIST t2_connector links NMTOKEN #REQUIRED>
        <!ATTLIST t2_connector composite NMTOKEN #REQUIRED>
        <!ELEMENT t2_prop EMPTY>
        <!ATTLIST t2_prop key NMTOKEN #REQUIRED>
        <!ATTLIST t2_prop value NMTOKEN #REQUIRED>
        <!ELEMENT t2_linked_prop EMPTY>
        <!ATTLIST t2_linked_prop target_id NMTOKEN #REQUIRED>
        <!ATTLIST t2_linked_prop key NMTOKEN #REQUIRED>
        <!ATTLIST t2_linked_prop value NMTOKEN #REQUIRED>
        <!ELEMENT t2_annot EMPTY>
        <!ELEMENT t2_display EMPTY>
        <!ATTLIST t2_display id NMTOKEN #REQUIRED>
        <!ATTLIST t2_display layer_id NMTOKEN #REQUIRED>
        <!ATTLIST t2_display x NMTOKEN #REQUIRED>
        <!ATTLIST t2_display y NMTOKEN #REQUIRED>
        <!ATTLIST t2_display magnification NMTOKEN #REQUIRED>
        <!ATTLIST t2_display srcrect_x NMTOKEN #REQUIRED>
        <!ATTLIST t2_display srcrect_y NMTOKEN #REQUIRED>
        <!ATTLIST t2_display srcrect_width NMTOKEN #REQUIRED>
        <!ATTLIST t2_display srcrect_height NMTOKEN #REQUIRED>
        <!ATTLIST t2_display scroll_step NMTOKEN #REQUIRED>
        <!ATTLIST t2_display c_alphas NMTOKEN #REQUIRED>
        <!ATTLIST t2_display c_alphas_state NMTOKEN #REQUIRED>
        <!ATTLIST t2_display filter_enabled NMTOKEN #REQUIRED>
        <!ATTLIST t2_display filter_min_max_enabled NMTOKEN #REQUIRED>
        <!ATTLIST t2_display filter_min NMTOKEN #REQUIRED>
        <!ATTLIST t2_display filter_max NMTOKEN #REQUIRED>
        <!ATTLIST t2_display filter_invert NMTOKEN #REQUIRED>
        <!ATTLIST t2_display filter_clahe_enabled NMTOKEN #REQUIRED>
        <!ATTLIST t2_display filter_clahe_block_size NMTOKEN #REQUIRED>
        <!ATTLIST t2_display filter_clahe_histogram_bins NMTOKEN #REQUIRED>
        <!ATTLIST t2_display filter_clahe_max_slope NMTOKEN #REQUIRED>
        <!ELEMENT ict_transform EMPTY>
        <!ATTLIST ict_transform class CDATA #REQUIRED>
        <!ATTLIST ict_transform data CDATA #REQUIRED>
        <!ELEMENT iict_transform EMPTY>
        <!ATTLIST iict_transform class CDATA #REQUIRED>
        <!ATTLIST iict_transform data CDATA #REQUIRED>
        <!ELEMENT ict_transform_list (ict_transform|iict_transform)*>
        <!ELEMENT iict_transform_list (iict_transform*)>
] >

<trakem2>
        <project
                id="0"
                title=""" + '"' + xml_file_name + '"' + """
                """ + 'unuid="{}"'.format(unuid) + """
                """ + 'mipmaps_folder="trakem2.{}/trakem2.mipmaps/'.format(unuid) + """
                storage_folder=""
                mipmaps_format="4"
                image_resizing_mode="Area downsampling"
                n_mipmap_threads="8"
                n_undo_steps="32"
                look_ahead_cache="0"
        >
        </project>
        <t2_layer_set
                oid="3"
                width="20.0"
                height="20.0"
                transform="matrix(1.0,0.0,0.0,1.0,0.0,0.0)"
                title="Top Level"
                links=""
                """ + 'layer_width="{}"'.format(float(layer_width)) + """
                """ + 'layer_height="{}"'.format(float(layer_height)) + """
                rot_x="0.0"
                rot_y="0.0"
                rot_z="0.0"
                snapshots_quality="true"
                snapshots_mode="Full"
                color_cues="true"
                area_color_cues="true"
                avoid_color_cue_colors="false"
                n_layers_color_cue="0"
                paint_arrows="true"
                paint_tags="true"
                paint_edge_confidence_boxes="true"
                prepaint="false"
                preload_ahead="0"
        >
                <t2_calibration
                        pixelWidth="4.0"
                        pixelHeight="4.0"
                        pixelDepth="30.0"
                        xOrigin="0.0"
                        yOrigin="0.0"
                        zOrigin="0.0"
                        info="null"
                        valueUnit="Gray Value"
                        timeUnit="sec"
                        unit="nm"
                />
                """
    out_data.writelines(lines)


def write_post_xml(out_data, layer_width, layer_height):
    lines = """</t2_layer_set>
            """ + '<t2_display id="{}"'.format(fetch_and_increase_oid()) + """
                layer_id="1"
                c_alphas="-1"
                c_alphas_state="-1"
                x="0"
                y="0"
                magnification="0.013888888888888888"
                srcrect_x="0"
                srcrect_y="0"
                """ + 'srcrect_width="{}"'.format(layer_width) + """
                """ + 'srcrect_height="{}"'.format(layer_height) + """
                scroll_step="1"
                filter_enabled="false"
                filter_min_max_enabled="false"
                filter_min="0"
                filter_max="255"
                filter_invert="false"
                filter_clahe_enabled="false"
                filter_clahe_block_size="127"
                filter_clahe_histogram_bins="256"
                filter_clahe_max_slope="3.0"
        />
</trakem2>
    """
    out_data.writelines(lines)

def write_layer(out_data, layer, layer_patches):
    pre_layer = '<t2_layer oid="{}"'.format(fetch_and_increase_oid()) + """
                         """ + 'thickness="7.5"' + """
                         """ + 'z="{}.0"'.format(layer) + """
                         title=""
                >
"""
    post_layer = "</t2_layer>\n"

    out_data.writelines(pre_layer)
    for tile in layer_patches:
        write_patch(out_data, tile)
    out_data.writelines(post_layer)


def write_patch(out_data, tile):
    patch_lines = \
"""                        <t2_patch
                                """ + 'oid="{}"'.format(fetch_and_increase_oid()) + """
                                """ + 'width="{}.0"'.format(tile["width"]) + """
                                """ + 'height="{}.0"'.format(tile["height"]) + """
                                """ + 'transform="matrix(1.0,0.0,0.0,1.0,{},{})"'.format(float(tile["tx"]), float(tile["ty"])) + """
                                """ + 'title="{}"'.format(tile["file_base_name"]) + """
                                links=""
                                type="0"
                                """ + 'file_path="{}"'.format(tile["file_full_path"]) + """
                                style="fill-opacity:1.0;stroke:#ffff00;"
                                """ + 'o_width="{}"'.format(tile["width"]) + """
                                """ + 'o_height="{}"'.format(tile["height"]) + """
                                min="0.0"
                                max="255.0"
                                mres="32"
                        >
                        </t2_patch>
"""
    out_data.writelines(patch_lines)


def parse_layer(base_dir, images, x, y):
    '''Writes the tilespec for a single section'''
    tiles = []
    layer_size = [0, 0]
    image_size = None

    for i, img in enumerate(images):
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
        tiles.append(tile)
        layer_size[0] = max(layer_size[0], image_size[0] + tile["ty"])
        layer_size[1] = max(layer_size[1], image_size[1] + tile["tx"])

    # if len(tiles) > 0:
    #     write_layer(out_data, layer, tiles)
    # else:
    #     print('Nothing to write from directory {}'.format(subdir))
    if len(tiles) == 0:
        print('Nothing to write from directory {}'.format(base_dir))
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
            img = row[0]
            # If not using windows, change the folder separator
            if os.name == "posix":
                img = img.replace('\\', '/')
            # Make sure that no duplicates appear
            if img not in images_dict.keys():
                images.append(img)
                images_dict[img] = True
                cur_x = float(row[1])
                cur_y = float(row[2])
                x.append(cur_x)
                y.append(cur_y)
    return images, x, y

def offset_list(lst):
    m = min(lst)
    return [item - m for item in lst]


if __name__ == '__main__':

    # Command line parser
    parser = argparse.ArgumentParser(description='Creates the TrakEM2 xml file for images that were acquired using the multiebeam EM.')
    parser.add_argument('input_folder', metavar='input_folder', type=str, 
                    help='a directory that contains sections subdirectories in which a full_coordinates txt file per section')
    parser.add_argument('-o', '--output_xml', type=str, 
                    help='the xml file where all the tiles will be stored (default: ./output.xml)',
                    default='./output.xml')
    # parser.add_argument('-v', '--overlap', type=float, 
    #                 help='the overlap fraction between the tiles (default: 0.06)', # 0.06 from Alyssa's data
    #                 default=0.06)

    args = parser.parse_args()

    xml_file_name = os.path.basename(args.output_xml)

    print("Parsing all subsections from folder: {}".format(args.input_folder))
    all_sub_folders = sorted(glob.glob(os.path.join(args.input_folder, '*')))


    max_layer_width = 0
    max_layer_height = 0

    all_layers = []

    for sub_folder in all_sub_folders:
        if os.path.isdir(sub_folder):
            print("Parsing subfolder: {}".format(sub_folder))
            coords_file = os.path.join(sub_folder, "full_image_coordinates.txt")
            if os.path.exists(coords_file):
                images, x, y = parse_coordinates_file(coords_file)
                x = offset_list(x)
                y = offset_list(y)
                cur_layer = parse_layer(sub_folder, images, x, y)
                max_layer_width = max(max_layer_width, cur_layer["width"])
                max_layer_height = max(max_layer_height, cur_layer["height"])
                layer = int(sub_folder.split(os.path.sep)[-1])
                cur_layer["layer_num"] = layer
                all_layers.append(cur_layer)
            else:
                print("Could not find full_image_coordinates.txt, skipping subfolder")

    max_layer_width = int(math.ceil(max_layer_width))
    max_layer_height = int(math.ceil(max_layer_height))

    print("Done parsing, writing xml file")

    # Create the output file and write the first lines of the xml
    out_data = open(args.output_xml, "w")
    write_pre_xml(out_data, xml_file_name, max_layer_width, max_layer_height)
    # write the tiles per layer
    for cur_layer in all_layers:
        write_layer(out_data, cur_layer["layer_num"], cur_layer["tiles"])

    # write the last lines of the xml
    write_post_xml(out_data, max_layer_width, max_layer_height)
    out_data.close()

