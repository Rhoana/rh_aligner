#
# Executes the rendering process jobs on the cluster (based on Rhoana's driver).
# It takes a collection of tilespec files, each describing a montage of a single section after 3d alignment,
# and renders the images.
# The input is a directory with tilespec files in json format (each file for a single layer),
# and a workspace directory where the intermediate and result files will be located.
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
from utils import path2url, create_dir, read_layer_from_file, write_list_to_file
from job import Job


class UpdateBoundingBox(Job):
    def __init__(self, tiles_fname, output_dir, jar_file, output_file, threads_num=1):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.output_dir = '-o "{0}"'.format(output_dir)
        self.jar_file = '-j "{0}"'.format(jar_file)
        self.dependencies = []
        self.threads = threads_num
        self.threads_str = "-t {0}".format(threads_num)
        self.memory = 10000
        self.time = 300
        self.is_java_job = True
        self.output = output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['RENDERER'], 'scripts', 'update_bounding_box.py'),
                self.output_dir, self.jar_file, self.threads_str, self.tiles_fname]


class NormalizeCoordinates(Job):
    def __init__(self, dependencies, tiles_fnames, output_dir, jar_file, outputs):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fnames = '{0}'.format(" ".join(tiles_fnames))
        self.output_dir = '-o "{0}"'.format(output_dir)
        self.jar_file = '-j "{0}"'.format(jar_file)
        self.dependencies = dependencies
        self.memory = 4000
        self.time = 100
        self.is_java_job = True
        self.output = outputs
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['RENDERER'], 'scripts', 'normalize_coordinates.py'),
                self.output_dir, self.jar_file, self.tiles_fnames]



class Render3D(Job):
    def __init__(self, dependencies, tiles_fname, output_dir, layer, scale, from_x, from_y, to_x, to_y, jar_file, output_file, threads_num=1):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.output_dir = '-o "{0}"'.format(output_dir)
        self.jar_file = '-j "{0}"'.format(jar_file)
        # Make the from-layer and to-layer the same layer
        self.from_layer = '--from_layer {0}'.format(layer)
        self.to_layer = '--to_layer {0}'.format(layer)
        self.scale = '--scale {0}'.format(scale)
        self.from_x = '--from_x {0}'.format(from_x)
        self.from_y = '--from_y {0}'.format(from_y)
        self.to_x = '--to_x {0}'.format(to_x)
        self.to_y = '--to_y {0}'.format(to_y)
        self.dependencies = dependencies
        self.threads = threads_num
        self.threads_str = "-t {0}".format(threads_num)
        self.memory = 33000
        self.time = 300
        self.is_java_job = True
        self.output = output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['RENDERER'], 'scripts', 'render_3d.py'),
                self.output_dir, self.jar_file, self.from_layer, self.to_layer, self.scale,
                self.from_x, self.from_y, self.to_x, self.to_y, self.threads_str, self.tiles_fname]




###############################
# Driver
###############################
if __name__ == '__main__':



    # Command line parser
    parser = argparse.ArgumentParser(description='Renders a given set of images using the SLURM cluster commands.')
    parser.add_argument('tiles_dir', metavar='tiles_dir', type=str, 
                        help='a directory that contains a tile_spec files in json format')
    parser.add_argument('-w', '--workspace_dir', type=str, 
                        help='a directory where the output files of the different stages will be kept (default: ./temp)',
                        default='./temp')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='the directory where the rendered output files will be stored (default: ./output)',
                        default='./output')
    parser.add_argument('-j', '--jar_file', type=str, 
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')
    parser.add_argument('--scale', type=float, 
                        help='set the scale of the rendered images (default: full image)',
                        default=1.0)
    parser.add_argument('--from_layer', type=int, 
                        help='the layer to start from (inclusive, default: the first layer in the data)',
                        default=-1)
    parser.add_argument('--to_layer', type=int, 
                        help='the last layer to render (inclusive, default: the last layer in the data)',
                        default=-1)
    parser.add_argument('--from_x', type=int, 
                        help='the left coordinate (default: 0)',
                        default=0)
    parser.add_argument('--from_y', type=int, 
                        help='the top coordinate (default: 0)',
                        default=0)
    parser.add_argument('--to_x', type=int, 
                        help='the right coordinate (default: full image)',
                        default=-1)
    parser.add_argument('--to_y', type=int, 
                        help='the bottom coordinate (default: full image)',
                        default=-1)
    parser.add_argument('-k', '--keeprunning', action='store_true', 
                        help='Run all jobs and report cluster jobs execution stats')
    parser.add_argument('-m', '--multicore', action='store_true', 
                        help='Run all jobs in blocks on multiple cores')
    parser.add_argument('-mk', '--multicore_keeprunning', action='store_true', 
                        help='Run all jobs in blocks on multiple cores and report cluster jobs execution stats')

    args = parser.parse_args() 

    assert 'RENDERER' in os.environ
    #assert 'VIRTUAL_ENV' in os.environ

    # create a workspace directory if not found
    create_dir(args.workspace_dir)

    bbox_dir = os.path.join(args.workspace_dir, "after_bbox")
    create_dir(bbox_dir)
    norm_dir = os.path.join(args.workspace_dir, "after_norm")
    create_dir(norm_dir)
    create_dir(args.output_dir)

    # Get all input json files (one per section) into a dictionary {json_fname -> [filtered json fname, sift features file, etc.]}
    json_files = dict((jf, {}) for jf in (glob.glob(os.path.join(args.tiles_dir, '*.json'))))

    bbox_files = []
    norm_files = []

    jobs = {}
    jobs['bbox'] = []
    norm_job = None
    jobs['render'] = []
    bbox_and_norm_jobs = []

    # Run the update bounding box process on each section (file)
    for f in json_files.keys():
        # Save the output files (after bbox and after normalization)
        tiles_fname = os.path.basename(f)
        bbox_out_file = os.path.join(bbox_dir, tiles_fname)
        bbox_files.append(bbox_out_file)

        norm_out_file = os.path.join(norm_dir, tiles_fname)
        norm_files.append(norm_out_file)

        # Create the job
        if not os.path.exists(bbox_out_file):
            bbox_job = UpdateBoundingBox(f, bbox_dir, args.jar_file, bbox_out_file, threads_num=8)
            jobs['bbox'].append(bbox_job)
            bbox_and_norm_jobs.append(bbox_job)


    # Normalize the coordination on all files (at a single execution)
    normalized_all_files = True
    for f in norm_files:
        if not os.path.exists(f):
            normalized_all_files = False
            break

    if not normalized_all_files:
        norm_job = NormalizeCoordinates(jobs['bbox'], bbox_files, norm_dir, args.jar_file, norm_files)
        bbox_and_norm_jobs.append(norm_job)

    norm_list_file = os.path.join(args.workspace_dir, "all_norm_files.txt")
    write_list_to_file(norm_list_file, norm_files)

    # Perform the rendering
    for f in json_files.keys():
        # read the layer from the file
        layer = read_layer_from_file(f)

        # If the layer in the file is not in the required range, continue to the next file
        if args.from_layer != -1:
            if layer < args.from_layer:
                continue
        if args.to_layer != -1:
            if layer > args.to_layer:
                continue

        tiles_fname = os.path.basename(f)
        # norm_file = os.path.join(norm_dir, tiles_fname)

        tiles_fname_prefix = os.path.splitext(tiles_fname)[0]
        render_out_file = os.path.join(args.output_dir, "{0:0>4}".format(layer) + "_" + tiles_fname_prefix + ".png")

        if not os.path.exists(render_out_file):
            # print "Adding job for output file: {0}".format(render_out_file)
            render_job = Render3D(bbox_and_norm_jobs, norm_list_file, args.output_dir, layer, args.scale,
                args.from_x, args.from_y, args.to_x, args.to_y, args.jar_file, render_out_file, threads_num=8)
            jobs['render'].append(render_job)


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

