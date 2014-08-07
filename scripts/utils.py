# Utils for the other python scripts

import os
import urlparse, urllib
from subprocess import call
import sys


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
        with open(args.conf_file_name, 'r') as conf_file:
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


