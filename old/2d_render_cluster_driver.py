#
# Executes the rendering jobs on the cluster (based on Rhoana's driver).
# It takes a collection of tilespec files, each describing a montage of a single section,
# normalizaes each section, and renders its 2d image.
# The input is a directory with tilespec files in json format (each file for a single layer),
# and a workspace directory where the intermediate and result files will be located.
# Renders the full images in full resolution.
#

import sys
import os.path
import os
import subprocess
import datetime
import time
from itertools import product
from collections import defaultdict
import argparse
import glob
import json
from utils import path2url, create_dir, read_layer_from_file, parse_range
from job import Job


class NormalizeCoordinates(Job):
    def __init__(self, tiles_fname, output_dir, output_file, jar_file):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.output_dir = '-o "{0}"'.format(output_dir)
        self.jar_file = '-j "{0}"'.format(jar_file)
        self.dependencies = []
        self.memory = 3000
        self.time = 10
        self.is_java_job = True
        self.output = output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'normalize_coordinates.py'),
                self.output_dir, self.jar_file, self.tiles_fname]


class Render2D(Job):
    def __init__(self, dependencies, tiles_fname, out_fname, width, jar_file, threads_num=1):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.output_file = '-o "{0}"'.format(out_fname)
        self.jar_file = '-j "{0}"'.format(jar_file)
        self.width = '-w {0}'.format(width)
        self.threads = threads_num
        self.threads_str = '-t {0}'.format(threads_num)
        self.dependencies = dependencies
        self.memory = 34000
        self.time = 500
        self.is_java_job = True
        self.output = out_fname
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'render_2d.py'),
                self.output_file, self.jar_file, self.width, self.threads_str, self.tiles_fname]





###############################
# Driver
###############################
if __name__ == '__main__':



    # Command line parser
    parser = argparse.ArgumentParser(description='Aligns (2d-elastic montaging) a given set of images using the SLURM cluster commands.')
    parser.add_argument('tiles_dir', metavar='tiles_dir', type=str, 
                        help='a directory that contains a tile_spec files in json format')
    parser.add_argument('-w', '--workspace_dir', type=str, 
                        help='a directory where the output files of the different stages will be kept (default: ./temp)',
                        default='./temp')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='the directory where the output to be rendered in json format files will be stored (default: ./output)',
                        default='./output')
    parser.add_argument('-j', '--jar_file', type=str, 
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')
    parser.add_argument('-f', '--format', type=str, 
                        help='the output format (default: png)',
                        default='png')
    parser.add_argument('-s', '--skip_layers', type=str, 
                        help='the range of layers (sections) that will not be processed e.g., "2,3,9-11,18" (default: no skipped sections)',
                        default=None)
    parser.add_argument('-k', '--keeprunning', action='store_true', 
                        help='Run all jobs and report cluster jobs execution stats')
    parser.add_argument('-m', '--multicore', action='store_true', 
                        help='Run all jobs in blocks on multiple cores')
    parser.add_argument('-mk', '--multicore_keeprunning', action='store_true', 
                        help='Run all jobs in blocks on multiple cores and report cluster jobs execution stats')


    args = parser.parse_args() 

    assert 'ALIGNER' in os.environ
    #assert 'VIRTUAL_ENV' in os.environ


    # create a workspace directory if not found
    create_dir(args.workspace_dir)


    norm_dir = os.path.join(args.workspace_dir, "normalized")
    create_dir(norm_dir)
    create_dir(args.output_dir)

    # Get all input json files (one per section) into a dictionary {json_fname -> [filtered json fname, sift features file, etc.]}
    json_files = dict((jf, {}) for jf in (glob.glob(os.path.join(args.tiles_dir, '*.json'))))


    skipped_layers = parse_range(args.skip_layers)



    for f in json_files.keys():
        tiles_fname_basename = os.path.basename(f)
        tiles_fname_prefix = os.path.splitext(tiles_fname_basename)[0]

        # read the layer from the file
        layer = read_layer_from_file(f)

        # Check if we need to skip the layer
        if layer in skipped_layers:
            print "Skipping layer {}".format(layer)
            continue

        slayer = str(layer)


        job_normalize = None
        job_render = None

        # Normalize the json file
        norm_json = os.path.join(norm_dir, tiles_fname_basename)
        if not os.path.exists(norm_json):
            print "Normalizing layer: {0}".format(slayer)
            job_normalize = NormalizeCoordinates(f, norm_dir, norm_json, args.jar_file)

        # Render the normalized json file
        out_fname = os.path.join(args.output_dir, "{0}.{1}".format(tiles_fname_prefix, args.format))
        if not os.path.exists(out_fname):
            dependencies = [ ]
            if job_normalize != None:
                dependencies.append(job_normalize)
            job_render = Render2D(dependencies, norm_json, out_fname, -1, args.jar_file, threads_num=8)





    # Run all jobs
    if args.keeprunning:
        Job.keep_running()
    elif args.multicore:
        # Bundle jobs for multicore nodes
        # if RUN_LOCAL:
        #     print "ERROR: --local cannot be used with --multicore (not yet implemented)."
        #     sys.exit(1)
        Job.multicore_run_all()
    elif args.multicore_keeprunning:
        # Bundle jobs for multicore nodes
        Job.multicore_keep_running()
    else:
        Job.run_all()

