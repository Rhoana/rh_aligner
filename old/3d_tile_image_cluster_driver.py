#
# Executes the tile image process jobs on the cluster (based on Rhoana's driver).
# It takes a collection of image files, each of a single section (after 3d alignment),
# and tiles these images.
# The input is a directory with image files (each file for a single layer),
# and each image tiles are going to be saved in a different directory (according to the image name) in the output directory.
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


class TileImage(Job):
    def __init__(self, image_file, output_dir, output_file, tile_size, output_pattern, output_type, processes_num=1):
        Job.__init__(self)
        self.already_done = False
        self.image_file = '"{0}"'.format(image_file)
        self.output_dir = '-o "{0}"'.format(output_dir)
        self.output_type = '--output_type "{0}"'.format(output_type)
        self.output_pattern = '--output_pattern "{0}"'.format(output_pattern)
        self.tile_size = '--tile_size "{0}"'.format(tile_size)
        self.dependencies = []
        self.threads = processes_num
        self.processes_str = "--processes_num {0}".format(processes_num)
        self.memory = 32000
        self.time = 300
        self.output = output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['RENDERER'], 'scripts', 'tile_image.py'),
                self.output_dir, self.output_type, self.output_pattern, self.tile_size, self.processes_str, self.image_file]




###############################
# Driver
###############################
if __name__ == '__main__':



    # Command line parser
    parser = argparse.ArgumentParser(description='Tiless a given set of images using the SLURM cluster commands.')
    parser.add_argument('images_dir', metavar='images_dir', type=str, 
                        help='a directory that contains all the image files to be tiled')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='the directory where the tiles image directories will be stored (default: ./output)',
                        default='./output')
    parser.add_argument('-p', '--processes_num', type=int,
                        help='Number of python processes to create the tiles (default: 1)',
                        default=1)
    parser.add_argument('-s', '--tile_size', type=int,
                        help='the size (square side) of each tile (default: 512)',
                        default=512)
    parser.add_argument('--output_type', type=str,
                        help='The output type format (default: png)',
                        default='png')
    parser.add_argument('-k', '--keeprunning', action='store_true', 
                        help='Run all jobs and report cluster jobs execution stats')
    parser.add_argument('-m', '--multicore', action='store_true', 
                        help='Run all jobs in blocks on multiple cores')
    parser.add_argument('-mk', '--multicore_keeprunning', action='store_true', 
                        help='Run all jobs in blocks on multiple cores and report cluster jobs execution stats')

    args = parser.parse_args() 

    assert 'RENDERER' in os.environ
    #assert 'VIRTUAL_ENV' in os.environ

    create_dir(args.output_dir)

    # Get all image files (one per section)
    image_files = glob.glob(os.path.join(args.images_dir, '*'))

    jobs = {}
    jobs['tile'] = []

    # Run the tiling process on each section (file)
    for f in image_files:
        # Get the output files
        image_fname = os.path.basename(f)
        image_out_dir = os.path.join(args.output_dir, os.path.splitext(image_fname)[0])

        # Create the job
        if not os.path.exists(image_out_dir):
            # need to give at least one dependency file
            output_pattern = "{}%rowcol".format(os.path.splitext(image_fname)[0])
            first_output_file = os.path.join(image_out_dir, "{}.{}".format(output_pattern.replace("%rowcol", "_tr1_tc1_"), args.output_type))
            print "Adding tile job of image: {}".format(f)
            tile_job = TileImage(f, image_out_dir, first_output_file, args.tile_size, output_pattern, args.output_type, args.processes_num)
            jobs['tile'].append(tile_job)


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

