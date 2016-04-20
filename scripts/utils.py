# Utils for the other python scripts

import os
import urlparse, urllib
import sys
import json
import time
import math

def path2url(path):
    return urlparse.urljoin('file:', urllib.pathname2url(os.path.abspath(path)))


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

def parse_range(s):
    result=set()
    if s is not None and len(s) != 0:
        for part in s.split(','):
            x = part.split('-')
            result.update(range(int(x[0]), int(x[-1]) + 1))
    return sorted(result)

def load_tilespecs(tile_file):
    tile_file = tile_file.replace('file://', '')
    with open(tile_file, 'r') as data_file:
        tilespecs = json.load(data_file)

    return tilespecs


