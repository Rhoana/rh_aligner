#
# Executes the alignment process jobs on the cluster (based on Rhoana's driver).
# It takes a collection of tilespec files, each describing a montage of a single section,
# and performs a 3d alignment of the entire json files.
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
from utils import path2url, create_dir, read_layer_from_file, parse_range
from job import Job


class CreateSiftFeatures(Job):
    def __init__(self, tiles_fname, output_file, jar_file, conf_fname=None, threads_num=1):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.output_file = '-o "{0}"'.format(output_file)
        self.jar_file = '-j "{0}"'.format(jar_file)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        self.dependencies = []
        self.threads = threads_num
        self.threads_str = "-t {0}".format(threads_num)
        self.memory = 16000
        self.time = 300
        self.is_java_job = True
        self.output = output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'create_sift_features.py'),
                self.output_file, self.jar_file, self.threads_str, self.conf_fname, self.tiles_fname]

class MatchSiftFeaturesAndFilter(Job):
    def __init__(self, dependencies, tiles_fname, features_fname, corr_output_file, jar_file, conf_fname=None):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.features_fname = '"{0}"'.format(features_fname)
        self.output_file = '-o "{0}"'.format(corr_output_file)
        self.jar_file = '-j "{0}"'.format(jar_file)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        self.dependencies = dependencies
        #self.threads = threads_num
        #self.threads_str = "-t {0}".format(threads_num)
        self.memory = 4000
        self.time = 240
        self.is_java_job = True
        self.output = corr_output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'match_sift_features_and_filter.py'),
                self.output_file, self.jar_file, self.conf_fname, self.tiles_fname, self.features_fname]

class OptimizeMontageTransform(Job):
    def __init__(self, dependencies, tiles_fname, corr_fname, fixed_tiles, opt_output_file, jar_file, conf_fname=None, threads_num=1):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.corr_fname = '"{0}"'.format(corr_fname)
        self.output_file = '-o "{0}"'.format(opt_output_file)
        self.jar_file = '-j "{0}"'.format(jar_file)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        if fixed_tiles is None:
            self.fixed_tiles = ''
        else:
            self.fixed_tiles = '-f {0}'.format(" ".join(str(f) for f in fixed_tiles))
        self.dependencies = dependencies
        #self.threads = threads_num
        #self.threads_str = "-t {0}".format(threads_num)
        self.memory = 6000
        self.time = 200
        self.is_java_job = True
        self.output = opt_output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'optimize_montage_transform.py'),
                self.output_file, self.fixed_tiles, self.jar_file, self.conf_fname, self.corr_fname, self.tiles_fname]



class MatchByMaxPMCC(Job):
    def __init__(self, dependencies, tiles_fname, fixed_tiles, pmcc_output_file, jar_file, conf_fname=None, threads_num=1):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.output_file = '-o "{0}"'.format(pmcc_output_file)
        self.jar_file = '-j "{0}"'.format(jar_file)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        if fixed_tiles is None:
            self.fixed_tiles = ''
        else:
            self.fixed_tiles = '-f {0}'.format(" ".join(str(f) for f in fixed_tiles))
        self.threads = threads_num
        self.threads_str = '-t {0}'.format(threads_num)
        self.dependencies = dependencies
        self.memory = 25000
        self.time = 600
        self.is_java_job = True
        self.output = pmcc_output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'match_by_max_pmcc.py'),
                self.output_file, self.fixed_tiles, self.jar_file, self.conf_fname, self.threads_str, self.tiles_fname]


class OptimizeMontageElastic(Job):
    def __init__(self, dependencies, tiles_fname, corr_fname, fixed_tiles, opt_output_file, jar_file, conf_fname=None, threads_num=1):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.corr_fname = '"{0}"'.format(corr_fname)
        self.output_file = '-o "{0}"'.format(opt_output_file)
        self.jar_file = '-j "{0}"'.format(jar_file)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        if fixed_tiles is None:
            self.fixed_tiles = ''
        else:
            self.fixed_tiles = '-f {0}'.format(" ".join(str(f) for f in fixed_tiles))
        self.threads = threads_num
        self.threads_str = '-t {0}'.format(threads_num)
        self.dependencies = dependencies
        self.memory = 10000
        self.time = 500
        self.is_java_job = True
        self.output = opt_output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'optimize_elastic_transform.py'),
                self.output_file, self.fixed_tiles, self.jar_file, self.conf_fname, self.corr_fname, self.tiles_fname]





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
    parser.add_argument('-c', '--conf_file_name', type=str, 
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)
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

    sifts_dir = os.path.join(args.workspace_dir, "sifts")
    create_dir(sifts_dir)
    matched_sifts_dir = os.path.join(args.workspace_dir, "matched_sifts")
    create_dir(matched_sifts_dir)
    opt_montage_dir = os.path.join(args.workspace_dir, "optimized_affine")
    create_dir(opt_montage_dir)
    matched_pmcc_dir = os.path.join(args.workspace_dir, "matched_pmcc")
    create_dir(matched_pmcc_dir)
    create_dir(args.output_dir)



    # Get all input json files (one per section) into a dictionary {json_fname -> [filtered json fname, sift features file, etc.]}
    json_files = dict((jf, {}) for jf in (glob.glob(os.path.join(args.tiles_dir, '*.json'))))


    skipped_layers = parse_range(args.skip_layers)



    all_layers = []
    jobs = {}
    layers_data = {}
    
    fixed_tile = 0

    for f in json_files.keys():
        tiles_fname_prefix = os.path.splitext(os.path.basename(f))[0]

        # read the layer from the file
        layer = read_layer_from_file(f)

        # Check if we need to skip the layer
        if layer in skipped_layers:
            print "Skipping layer {}".format(layer)
            continue

        slayer = str(layer)

        if not (slayer in layers_data.keys()):
            layers_data[slayer] = {}
            jobs[slayer] = {}
            jobs[slayer]['sifts'] = []


        all_layers.append(layer)


        job_sift = None
        job_match = None
        job_opt_montage = None
        job_pmcc = None

        # create the sift features of these tiles
        sifts_json = os.path.join(sifts_dir, "{0}_sifts.json".format(tiles_fname_prefix))
        if not os.path.exists(sifts_json):
            print "Computing layer sifts: {0}".format(slayer)
            job_sift = CreateSiftFeatures(f, sifts_json, args.jar_file, conf_fname=args.conf_file_name, threads_num=4)
            jobs[slayer]['sifts'].append(job_sift)


        layers_data[slayer]['ts'] = f
        layers_data[slayer]['sifts'] = sifts_json
        layers_data[slayer]['prefix'] = tiles_fname_prefix

        # match the sift features
        match_json = os.path.join(matched_sifts_dir, "{0}_sift_matches.json".format(tiles_fname_prefix))
        if not os.path.exists(match_json):
            print "Matching layer sifts: {0}".format(slayer)
            dependencies = [ ]
            if job_sift != None:
                dependencies.append(job_sift)
            job_match = MatchSiftFeaturesAndFilter(dependencies, layers_data[slayer]['ts'], \
                layers_data[slayer]['sifts'], match_json, \
                args.jar_file, conf_fname=args.conf_file_name)
        layers_data[slayer]['matched_sifts'] = match_json


        # optimize the matches (affine)
        opt_montage_json = os.path.join(opt_montage_dir, "{0}_opt_montage.json".format(tiles_fname_prefix))
        if not os.path.exists(opt_montage_json):
            print "Optimizing (affine) layer matches: {0}".format(slayer)
            dependencies = [ ]
            if job_sift != None:
                dependencies.append(job_sift)
            if job_match != None:
                dependencies.append(job_match)
            job_opt_montage = OptimizeMontageTransform(dependencies, layers_data[slayer]['ts'], \
                layers_data[slayer]['matched_sifts'], [ fixed_tile ], opt_montage_json, \
                args.jar_file, conf_fname=args.conf_file_name)
        layers_data[slayer]['optimized_montage'] = opt_montage_json


        # Match by max PMCC
        pmcc_fname = os.path.join(matched_pmcc_dir, "{0}_match_pmcc.json".format(tiles_fname_prefix))
        if not os.path.exists(pmcc_fname):
            print "Matching layers by Max PMCC: {0}".format(slayer)
            dependencies = [ ]
            if job_sift != None:
                dependencies.append(job_sift)
            if job_match != None:
                dependencies.append(job_match)
            if job_opt_montage != None:
                dependencies.append(job_opt_montage)
            job_pmcc = MatchByMaxPMCC(dependencies, layers_data[slayer]['optimized_montage'], \
                [ fixed_tile ], pmcc_fname, args.jar_file, conf_fname=args.conf_file_name, threads_num=8)
        layers_data[slayer]['matched_pmcc'] = pmcc_fname


        # Optimize (elastic) the 2d image
        opt_elastic_json = os.path.join(args.output_dir, "{0}.json".format(tiles_fname_prefix))
        if not os.path.exists(opt_elastic_json):
            print "Optimizing (elastic) layer matches: {0}".format(slayer)
            dependencies = [ ]
            if job_sift != None:
                dependencies.append(job_sift)
            if job_match != None:
                dependencies.append(job_match)
            if job_opt_montage != None:
                dependencies.append(job_opt_montage)
            if job_pmcc != None:
                dependencies.append(job_pmcc)
            job_opt_elastic = OptimizeMontageElastic(dependencies, layers_data[slayer]['optimized_montage'], \
                layers_data[slayer]['matched_pmcc'], [ fixed_tile ], opt_elastic_json, \
                args.jar_file, conf_fname=args.conf_file_name)






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

