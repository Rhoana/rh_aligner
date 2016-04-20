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
from utils import write_list_to_file, create_dir, read_layer_from_file, parse_range
from job import Job
from rh_aligner.common.bounding_box import BoundingBox





class PreliminaryMatchLayersMfovs(Job):
    def __init__(self, tiles_fname1, features_dir1, tiles_fname2, features_dir2, output_fname, conf_fname=None, threads_num=1):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname1 = '"{0}"'.format(tiles_fname1)
        self.features_dir1 = '"{0}"'.format(features_dir1)
        self.tiles_fname2 = '"{0}"'.format(tiles_fname2)
        self.features_dir2 = '"{0}"'.format(features_dir2)
        self.output_fname = '-o "{0}"'.format(output_fname)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        self.dependencies = [ ]
        #self.threads = threads_num
        #self.threads_str = "-t {0}".format(threads_num)
        self.memory = 1000
        self.time = 300
        self.output = output_fname
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'wrappers', 'pre_match_3d_incremental.py'),
                self.output_fname, self.conf_fname, self.tiles_fname1, self.features_dir1, self.tiles_fname2, self.features_dir2]


class MatchLayersByMaxPMCCMfov(Job):
    def __init__(self, dependencies, tiles_fname1, tiles_fname2, pre_match_fname, output_fname, targeted_mfov, conf_fname=None, threads_num=1, auto_add_model=False):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname1 = '"{0}"'.format(tiles_fname1)
        self.tiles_fname2 = '"{0}"'.format(tiles_fname2)
        self.pre_match_fname = '"{0}"'.format(pre_match_fname)
        self.targeted_mfov = '{0}'.format(targeted_mfov)
        self.output_fname = '-o "{0}"'.format(output_fname)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        #if auto_add_model:
        #    self.auto_add_model = '--auto_add_model'
        #else:
        #    self.auto_add_model = ''
        self.threads = threads_num
        self.threads_str = '-t {0}'.format(threads_num)
        self.dependencies = dependencies
        self.memory = 10000
        self.time = 800
        self.output = output_fname
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'wrappers', 'block_match_3d_multiprocess.py'),
                self.output_fname, self.conf_fname,
                self.threads_str, #self.auto_add_model,
                self.tiles_fname1, self.tiles_fname2, self.pre_match_fname, self.targeted_mfov]


class OptimizeLayersElastic(Job):
    def __init__(self, dependencies, outputs, tiles_fnames, corr_fnames, output_dir, max_layer_distance, conf_fname=None, skip_layers=None, threads_num=1):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fnames = '--tile_files {0}'.format(" ".join(tiles_fnames))
        self.corr_fnames = '--corr_files {0}'.format(" ".join(corr_fnames))
        self.output_dir = '-o "{0}"'.format(output_dir)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        if skip_layers is None:
            self.skip_layers = ''
        else:
            self.skip_layers = '-s "{0}"'.format(skip_layers)
        self.max_layer_distance = '-d {}'.format(max_layer_distance)
        self.threads = threads_num
        self.threads_str = '-t {0}'.format(threads_num)
        self.dependencies = dependencies
        self.memory = 32000
        self.time = 800
        self.output = outputs
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'wrappers', 'optimize_layers_elastic.py'),
                self.output_dir, self.conf_fname, self.threads_str,
                self.max_layer_distance, self.skip_layers, self.tiles_fnames, self.corr_fnames]





###############################
# Driver
###############################
if __name__ == '__main__':



    # Command line parser
    parser = argparse.ArgumentParser(description='Aligns a given set of images using the SLURM cluster commands.')
    parser.add_argument('tiles_dir', metavar='tiles_dir', type=str, 
                        help='a directory that contains a tile_spec files in json format')
    parser.add_argument('work_dir_2d', metavar='work_dir_2d', type=str, 
                        help='a directory where the 2d alignment work directory is located (includes the sift features)')
    parser.add_argument('-w', '--workspace_dir', type=str, 
                        help='a directory where the output files of the different stages will be kept (default: ./temp)',
                        default='./temp')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='the directory where the output to be rendered in json format files will be stored (default: ./output)',
                        default='./output')
    parser.add_argument('-c', '--conf_file_name', type=str, 
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)
    parser.add_argument('-d', '--max_layer_distance', type=int, 
                        help='the largest distance between two layers to be matched (default: 1)',
                        default=1)
    parser.add_argument('--auto_add_model', action="store_true", 
                        help='automatically add the identity model, if a model is not found')
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

    pre_matches_dir = os.path.join(args.workspace_dir, "pre_matched_mfovs")
    create_dir(pre_matches_dir)
    matched_pmcc_dir = os.path.join(args.workspace_dir, "matched_pmcc")
    create_dir(matched_pmcc_dir)



    # Get all input json files (one per section) into a dictionary {json_fname -> [filtered json fname, sift features file, etc.]}
    json_files = dict((jf, {}) for jf in (glob.glob(os.path.join(args.tiles_dir, '*.json'))))



    skipped_layers = parse_range(args.skip_layers)


    all_layers = []
    all_ts_files = []
    jobs = {}
    layers_data = {}

    all_running_jobs = []

    mfovs_per_layer = {}


    # Make sure we only parse the relevant sections
    for f in json_files.keys():
        # read the layer from the file
        data = None
        with open(f, 'r') as data_file:
            data = json.load(data_file)
        layer = data[0]['layer']

        if layer in skipped_layers:
            continue

        slayer = str(layer)
        # read the possible mfovs for the layer
        mfovs_per_layer[slayer] = set([tile["mfov"] for tile in data])

        if not (slayer in layers_data.keys()):
            layers_data[slayer] = {}
            jobs[slayer] = {}


        all_layers.append(layer)

        layers_data[slayer]['ts'] = f
        tiles_fname_prefix = os.path.splitext(os.path.basename(f))[0]
        layers_data[slayer]['prefix'] = tiles_fname_prefix
        layers_data[slayer]['sifts_dir'] = os.path.join(os.path.join(args.work_dir_2d, 'sifts'), tiles_fname_prefix.split("_montaged")[0])
        if not os.path.exists(layers_data[slayer]['sifts_dir']):
            print("Error: missing sifts directory for layer {}. Assuming it should be at: {}".format(slayer, layers_data[slayer]['sifts_dir']))
            sys.exit(1)
        all_ts_files.append(f)


    # Verify that all the layers are there and that there are no holes
    all_layers.sort()
    for i in range(len(all_layers) - 1):
        slayer = str(all_layers[i])
        layers_data[slayer]['pre_matched_mfovs'] = {}
        #layers_data[slayer]['matched_pmcc'] = {}
        if all_layers[i + 1] - all_layers[i] != 1:
            for l in range(all_layers[i] + 1, all_layers[i + 1]):
                if l not in skipped_layers:
                    print "Error missing layer {} between: {} and {}".format(l, all_layers[i], all_layers[i + 1])
                    sys.exit(1)
    slayer_last = str(all_layers[-1])
    layers_data[slayer_last]['pre_matched_mfovs'] = {}
    #layers_data[slayer_last]['matched_pmcc'] = {}

    print "Found the following layers: {0}".format(all_layers)

    # Match each two layers in the required distance
    all_pmcc_files = []
    pmcc_jobs = []
    for layer1_ind, layer1 in enumerate(all_layers):
        slayer1 = str(layer1)
        # Process all matched layers
        matched_after_layers = 0
        j = 1
        while matched_after_layers < args.max_layer_distance:
            if layer1 + j > all_layers[-1]:
                break

            layer2 = layer1 + j
            slayer2 = str(layer2)
            if layer1 in skipped_layers or layer2 in skipped_layers:
                print "Skipping matching of layers {} and {}, because at least one of them should be skipped".format(layer1, layer2)
                j += 1
                continue

            fname1_prefix = layers_data[slayer1]['prefix']
            fname2_prefix = layers_data[slayer2]['prefix']

            job_pre_match = None
            job_pmcc = None

            # match the features of neighboring tiles
            pre_match_json = os.path.join(pre_matches_dir, "{0}_{1}_pre_matches.json".format(fname1_prefix, fname2_prefix))
            if not os.path.exists(pre_match_json):
                print "Pre-Matching layers: {0} and {1}".format(layer1, layer2)
                job_pre_match = PreliminaryMatchLayersMfovs(layers_data[slayer1]['ts'], layers_data[slayer1]['sifts_dir'],
                    layers_data[slayer2]['ts'], layers_data[slayer2]['sifts_dir'], pre_match_json,
                    conf_fname=args.conf_file_name)
                all_running_jobs.append(job_pre_match)
            layers_data[slayer1]['pre_matched_mfovs'][slayer2] = pre_match_json


            # match by max PMCC the two layers (mfov after mfov)
            for mfov1 in mfovs_per_layer[slayer1]:
                pmcc_fname_mfov1 = os.path.join(matched_pmcc_dir, "{0}_{1}_match_pmcc_mfov_{2}.json".format(fname1_prefix, fname2_prefix, mfov1))
                if not os.path.exists(pmcc_fname_mfov1):
                    print "Matching layers by Max PMCC: {0} (mfov {1}) and {2}".format(i, mfov1, i + j)
                    dependencies = [ ]
                    if job_pre_match != None:
                        dependencies.append(job_pre_match)

                    job_pmcc_mfov1 = MatchLayersByMaxPMCCMfov(dependencies, layers_data[slayer1]['ts'], layers_data[slayer2]['ts'], 
                        layers_data[slayer1]['pre_matched_mfovs'][slayer2], 
                        pmcc_fname_mfov1, int(mfov1),
                        conf_fname=args.conf_file_name, threads_num=4, auto_add_model=args.auto_add_model)
                    all_running_jobs.append(job_pmcc_mfov1)
                    pmcc_jobs.append(job_pmcc_mfov1)
                #layers_data[slayer1]['matched_pmcc'][slayer2] = pmcc_fname_mfov1
                all_pmcc_files.append(pmcc_fname_mfov1)


            for mfov2 in mfovs_per_layer[slayer2]:
                pmcc_fname2_mfov2 = os.path.join(matched_pmcc_dir, "{0}_{1}_match_pmcc_mfov_{2}.json".format(fname2_prefix, fname1_prefix, mfov2))
                if not os.path.exists(pmcc_fname2_mfov2):
                    print "Matching layers by Max PMCC: {0} (mfov {1}) and {2}".format(i + j, mfov2, i)
                    dependencies = [ ]
                    if job_pre_match != None:
                        dependencies.append(job_pre_match)

                    job_pmcc2_mfov2 = MatchLayersByMaxPMCCMfov(dependencies, layers_data[slayer2]['ts'], layers_data[slayer1]['ts'], 
                        layers_data[slayer1]['pre_matched_mfovs'][slayer2], 
                        pmcc_fname2_mfov2, int(mfov2),
                        conf_fname=args.conf_file_name, threads_num=4, auto_add_model=args.auto_add_model)
                    pmcc_jobs.append(job_pmcc2_mfov2)
                    all_running_jobs.append(job_pmcc2_mfov2)
                #layers_data[slayer2]['matched_pmcc'][slayer1] = pmcc_fname2_mfov2
                all_pmcc_files.append(pmcc_fname2_mfov2)


            # Merge the multiple mfovs pmcc match files into one per direction
            pmcc_fname = os.path.join(matched_pmcc_dir, "{0}_{1}_match_pmcc.json".format(fname1_prefix, fname2_prefix))



            j += 1
            matched_after_layers += 1



    print "all_pmcc_files: {0}".format(all_pmcc_files)

    # Create a single file that lists all tilespecs and a single file that lists all pmcc matches (the os doesn't support a very long list)
    ts_list_file = os.path.join(args.workspace_dir, "all_ts_files.txt")
    write_list_to_file(ts_list_file, all_ts_files)
    pmcc_list_file = os.path.join(args.workspace_dir, "all_pmcc_files.txt")
    write_list_to_file(pmcc_list_file, all_pmcc_files)


    # Optimize all layers to a single 3d image
    create_dir(args.output_dir)
    sections_outputs = []
    for i in all_layers:
        out_section = os.path.join(args.output_dir, os.path.basename(layers_data[str(i)]['ts']))
        sections_outputs.append(out_section)

    dependencies = all_running_jobs
    job_optimize = OptimizeLayersElastic(dependencies, sections_outputs, [ ts_list_file ], [ pmcc_list_file ],
        args.output_dir, args.max_layer_distance, conf_fname=args.conf_file_name,
        skip_layers=args.skip_layers, threads_num=4)


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

