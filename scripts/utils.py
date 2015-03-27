# Utils for the other python scripts

import os
import urlparse, urllib
from subprocess import call
import sys
import json
import multiprocessing

def path2url(path):
    if "://" in path:
        return path
    return urlparse.urljoin('file:', urllib.pathname2url(os.path.abspath(path)))

def read_conf_args(conf_fname, tool):
    ''' Read the tool configuration from conf (json format), and return them as dictionary '''
    tool_dict = {}
    if not conf_fname is None:
        with open(conf_fname, 'r') as conf_file:
            conf = json.load(conf_file)
            if tool in conf:
                tool_dict = conf[tool]
    return tool_dict


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

def get_gc_threads_num(app_threads_num):
    if app_threads_num is None:
        app_threads_num = multiprocessing.cpu_count()

    gc_threads_num = 1
    if app_threads_num >= 16:
        gc_threads_num = 4
    elif app_threads_num >= 8:
        gc_threads_num = 3
    elif app_threads_num >= 4:
        gc_threads_num = 2

    return gc_threads_num
