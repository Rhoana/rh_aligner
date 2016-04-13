# Utils for the other python scripts

import os
import urlparse, urllib
from subprocess import call
import sys
import json
import time
import math

def path2url(path):
    return urlparse.urljoin('file:', urllib.pathname2url(os.path.abspath(path)))


def conf_args(conf, tool):
    ''' Read the tool configuration from conf (json format), and return the parameters in a string format '''
    res = ''
    if not conf is None:
        if tool in conf:
            tool_keys = conf[tool].keys()
            for tool_key in conf[tool]:
                res = res + "--{0} {1} ".format(tool_key, conf[tool][tool_key])
    return res

def conf_args_from_file(conf_fname, tool):
    ''' Read the tool configuration from conf file name (json format), and return the parameters in a string format '''
    res = ''
    if not conf_fname is None:
        with open(conf_fname, 'r') as conf_file:
            conf = json.load(conf_file)
            if tool in conf:
                tool_keys = conf[tool].keys()
                for tool_key in conf[tool]:
                    res = res + "--{0} {1} ".format(tool_key, conf[tool][tool_key])
    return res

def conf_from_file(conf_fname, tool):
    ''' Read the tool configuration from conf file name (json format), and return the parameters in a dictionary format '''
    res = None
    if not conf_fname is None:
        with open(conf_fname, 'r') as conf_file:
            conf = json.load(conf_file)
            if tool in conf:
                return conf[tool]
    return res

def execute_shell_command(cmd):
    print "Executing: {0}".format(cmd)
    res = call(cmd, shell=True) # w/o shell=True it seems that the env-vars are not set
    if res != 0:
        print "Error while executing: {0}".format(cmd)
        print "Exiting"
        sys.exit(1)


def create_dir(path):
    # create a directory if not found
    if not os.path.exists(path):
        os.makedirs(path)

def write_list_to_file(file_name, lst):
    with open(file_name, 'w') as out_file:
        for item in lst:
            out_file.write("%s\n" % path2url(item))

def read_layer_from_file(tiles_spec_fname):
    layer = None
    with open(tiles_spec_fname, 'r') as data_file:
        data = json.load(data_file)
    for tile in data:
        if tile['layer'] is None:
            print "Error reading layer in one of the tiles in: {0}".format(tiles_spec_fname)
            sys.exit(1)
        if layer is None:
            layer = tile['layer']
        if layer != tile['layer']:
            print "Error when reading tiles from {0} found inconsistent layers numbers: {1} and {2}".format(tiles_spec_fname, layer, tile['layer'])
            sys.exit(1)
    if layer is None:
        print "Error reading layers file: {0}. No layers found.".format(tiles_spec_fname)
        sys.exit(1)
    return int(layer)

def parse_range(s):
    result=set()
    if s is not None and len(s) != 0:
        for part in s.split(','):
            x = part.split('-')
            result.update(range(int(x[0]), int(x[-1]) + 1))
    return sorted(result)

def wait_after_file(filename, timeout_seconds):
    if timeout_seconds > 0:
        cur_time = time.time()
        mod_time = os.path.getmtime(filename)
        end_wait_time = mod_time + timeout_seconds
        while cur_time < end_wait_time:
            print "Waiting for file: {}".format(filename)
            cur_time = time.time()
            mod_time = os.path.getmtime(filename)
            end_wait_time = mod_time + timeout_seconds
            if cur_time < end_wait_time:
                time.sleep(end_wait_time - cur_time)

def load_tilespecs(tile_file):
    tile_file = tile_file.replace('file://', '')
    with open(tile_file, 'r') as data_file:
        tilespecs = json.load(data_file)

    return tilespecs

def index_tilespec(tilespec):
    """Given a section tilespec returns a dictionary of [mfov][tile_index] to the tile's tilespec"""
    index = {}
    for ts in tilespec:
        #mfov = str(ts["mfov"])
        mfov = ts["mfov"]
        if mfov not in index.keys():
            index[mfov] = {}
        #index[mfov][str(ts["tile_index"])] = ts
        index[mfov][ts["tile_index"]] = ts
    return index

def generate_hexagonal_grid(boundingbox, spacing):
    """Generates an hexagonal grid inside a given bounding-box with a given spacing between the vertices"""
    hexheight = spacing
    hexwidth = math.sqrt(3) * spacing / 2
    vertspacing = 0.75 * hexheight
    horizspacing = hexwidth
    sizex = int((boundingbox[1] - boundingbox[0]) / horizspacing) + 2
    sizey = int((boundingbox[3] - boundingbox[2]) / vertspacing) + 2
    if sizey % 2 == 0:
        sizey += 1
    pointsret = []
    for i in range(-2, sizex):
        for j in range(-2, sizey):
            xpos = i * spacing
            ypos = j * spacing
            if j % 2 == 1:
                xpos += spacing * 0.5
            if (j % 2 == 1) and (i == sizex - 1):
                continue
            pointsret.append([int(xpos + boundingbox[0]), int(ypos + boundingbox[2])])
    return pointsret

