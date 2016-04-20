#
# Executes the alignment process jobs on the cluster (based on Rhoana's driver).
# It takes a collection of tilespec files, each describing a montage of a single section,
# and performs a 2d affine alignment of the entire json files.
# The input is a directory with tilespec files in json format (each file for a single layer),
# and a workspace directory where the intermediate and result files will be located.
#

import sys
import os.path
import os
import subprocess
import datetime
import time
import itertools
import argparse
import glob
import json
from utils import create_dir, read_layer_from_file, parse_range, load_tilespecs, write_list_to_file
from rh_aligner.common.bounding_box import BoundingBox
from job import Job


class CreateSiftFeatures(Job):
    def __init__(self, tiles_fname, output_file, tile_index, conf_fname=None, threads_num=1):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.tile_index = '{0}'.format(tile_index)
        self.output_file = '-o "{0}"'.format(output_file)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        self.dependencies = []
        # self.threads = threads_num
        # self.threads_str = "-t {0}".format(threads_num)
        self.memory = 4000
        self.time = 20
        self.output = output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'wrappers', 'create_sift_features_cv2.py'),
                self.output_file, self.conf_fname, self.tiles_fname, self.tile_index]

class CreateMultipleSiftFeatures(Job):
    def __init__(self, tiles_fname, sifts_work_dir, temp_output_list_file, conf_fname=None, threads_num=1):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.tile_indices_list = []
        self.output_files_list = []
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        self.dependencies = []
        # self.threads = threads_num
        # self.threads_str = "-t {0}".format(threads_num)
        self.memory = 3000
        self.time = 100
        self.sifts_work_dir = sifts_work_dir
        self.temp_output_list_file = temp_output_list_file

    def add_job(self, output_file, tile_index):
        self.output_files_list.append(output_file)
        self.tile_indices_list.append(tile_index)
        self.output.append(output_file)

    def prepare_files(self):
        if len(self.tile_indices_list) > 0:
            self.tile_indices = '-i {0}'.format(' '.join([str(i) for i in self.tile_indices_list]))
            self.output_files = '-o {0}'.format(' '.join(self.output_files_list))
            #tmp_output_files = os.path.join(self.sifts_work_dir, "{}_sifts_outputs_lst.txt".format(self.temp_output_list_file))
            #with open(tmp_output_files, 'w') as f:
            #    for i, item in zip(self.tile_indices_list, self.output_files_list):
            #        f.write("{},{}\n".format(i, item))
            #self.output_files = '-o "{0}"'.format(tmp_output_files)



    def command(self):
        self.prepare_files()
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'wrappers', 'create_sift_features_cv2.py'),
                self.output_files, self.tile_indices, self.conf_fname, self.tiles_fname]




class MatchSiftFeaturesAndFilter(Job):
    def __init__(self, dependencies, tiles_fname, features_fname1, features_fname2, corr_output_file, index_pair, wait_time=None, conf_fname=None):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.features_fname1 = '"{0}"'.format(features_fname1)
        self.features_fname2 = '"{0}"'.format(features_fname2)
        self.index_pair = ':'.join(index_pair)
        self.output_file = '-o "{0}"'.format(corr_output_file)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        if wait_time is None:
            self.wait_time = ''
        else:
            self.wait_time = '-w {0}'.format(wait_time)
        self.dependencies = dependencies
        self.memory = 400
        self.time = 20
        self.output = corr_output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'wrappers', 'match_sift_features_and_filter_cv2.py'),
                self.output_file, self.wait_time, self.conf_fname,
                self.tiles_fname, self.features_fname1, self.features_fname2, self.index_pair]

class MatchMultipleSiftFeaturesAndFilter(Job):
    def __init__(self, matches_work_dir, tiles_fname, temp_files_prefix, threads_num=1, wait_time=None, conf_fname=None):
        Job.__init__(self)
        self.already_done = False
        self.dependencies = []
        self.features_fnames1_list = []
        self.features_fnames2_list = []
        self.output_files_list = []
        self.index_pairs_list = []
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        if wait_time is None:
            self.wait_time = ''
        else:
            self.wait_time = '-w {0}'.format(wait_time)
        self.memory = 3000
        self.time = 100
        self.threads = threads_num
        self.threads_str = "-t {0}".format(threads_num)
        self.matches_work_dir = matches_work_dir
        self.temp_files_prefix = temp_files_prefix

    def add_job(self, dependencies, features_fname1, features_fname2, corr_output_file, index_pair):
        self.features_fnames1_list.append(features_fname1)
        self.features_fnames2_list.append(features_fname2)
        self.output_files_list.append(corr_output_file)
        self.index_pairs_list.append(index_pair)
        self.dependencies.extend(dependencies)
        self.output.append(corr_output_file)


    def prepare_files(self):
        if len(self.features_fnames1_list) > 0:
            tmp_file_features1 = os.path.join(self.matches_work_dir, "{}_features_lst1.txt".format(self.temp_files_prefix))
            tmp_file_features2 = os.path.join(self.matches_work_dir, "{}_features_lst2.txt".format(self.temp_files_prefix))
            tmp_output_files = os.path.join(self.matches_work_dir, "{}_outputs_lst.txt".format(self.temp_files_prefix))
            with open(tmp_file_features1, 'w') as f:
                for item in self.features_fnames1_list:
                    f.write("{}\n".format(item))
            with open(tmp_file_features2, 'w') as f:
                for item in self.features_fnames2_list:
                    f.write("{}\n".format(item))
            with open(tmp_output_files, 'w') as f:
                for item in self.output_files_list:
                    f.write("{}\n".format(item))
            self.features_fnames1 = '"{0}"'.format(tmp_file_features1)
            self.features_fnames2 = '"{0}"'.format(tmp_file_features2)
            self.index_pairs = '--index_pairs {}'.format(' '.join([':'.join(index_pair) for index_pair in self.index_pairs_list]))
            self.output_files = '-o "{0}"'.format(tmp_output_files)

    def command(self):
        self.prepare_files()
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'wrappers', 'match_sift_features_and_filter_cv2.py'),
                self.output_files, self.wait_time, self.conf_fname, self.threads_str,
                self.tiles_fname, self.features_fnames1, self.features_fnames2, self.index_pairs]



class OptimizeMontageTransform(Job):
    def __init__(self, dependencies, tiles_fname, matches_list_file, opt_output_file, conf_fname=None, threads_num=1):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.matches_list_file = '"{0}"'.format(matches_list_file)
        self.output_file = '-o "{0}"'.format(opt_output_file)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        # if fixed_tiles is None:
        #     self.fixed_tiles = ''
        # else:
        #     self.fixed_tiles = '-f {0}'.format(" ".join(str(f) for f in fixed_tiles))
        self.dependencies = dependencies
        # self.threads = threads_num
        # self.threads_str = "-t {0}".format(threads_num)
        self.memory = 6000
        self.time = 600
        self.output = opt_output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'wrappers', 'optimize_2d_mfovs.py'),
                self.output_file, self.conf_fname, self.tiles_fname, self.matches_list_file]




###############################
# Driver
###############################
if __name__ == '__main__':



    # Command line parser
    parser = argparse.ArgumentParser(description='Aligns (2d-elastic montaging) a given set of multibeam images using the SLURM cluster commands.')
    parser.add_argument('tiles_dir', metavar='tiles_dir', type=str, 
                        help='a directory that contains tile_spec files in json format')
    parser.add_argument('-w', '--workspace_dir', type=str, 
                        help='a directory where the output files of the different stages will be kept (default: ./temp)',
                        default='./temp')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='the directory where the output to be rendered in json format files will be stored (default: ./output)',
                        default='./output')
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
    create_dir(args.output_dir)



    # Get all input json files (one per section) into a dictionary {json_fname -> [filtered json fname, sift features file, etc.]}
    json_files = dict((jf, {}) for jf in (glob.glob(os.path.join(args.tiles_dir, '*.json'))))


    skipped_layers = parse_range(args.skip_layers)



    all_layers = []
    jobs = {}
    layers_data = {}
    
    fixed_tile = 0

    for f in sorted(json_files.keys()):
        tiles_fname_prefix = os.path.splitext(os.path.basename(f))[0]

        cur_tilespec = load_tilespecs(f)

        # read the layer from the file
        layer = None
        for tile in cur_tilespec:
            if tile['layer'] is None:
                print "Error reading layer in one of the tiles in: {0}".format(f)
                sys.exit(1)
            if layer is None:
                layer = int(tile['layer'])
            if layer != tile['layer']:
                print "Error when reading tiles from {0} found inconsistent layers numbers: {1} and {2}".format(f, layer, tile['layer'])
                sys.exit(1)
        if layer is None:
            print "Error reading layers file: {0}. No layers found.".format(f)
            continue

        # Check if we need to skip the layer
        if layer in skipped_layers:
            print "Skipping layer {}".format(layer)
            continue

        slayer = str(layer)

        if not (slayer in layers_data.keys()):
            layers_data[slayer] = {}
            jobs[slayer] = {}
            jobs[slayer]['sifts'] = {}
            jobs[slayer]['matched_sifts'] = {}
            jobs[slayer]['matched_sifts']['intra'] = {}
            jobs[slayer]['matched_sifts']['inter'] = None
            layers_data[slayer]['ts'] = f
            layers_data[slayer]['sifts'] = {}
            layers_data[slayer]['prefix'] = tiles_fname_prefix
            layers_data[slayer]['matched_sifts'] = []

        layer_sifts_dir = os.path.join(sifts_dir, layers_data[slayer]['prefix'])
        layer_matched_sifts_intra_dir = os.path.join(matched_sifts_dir, os.path.join(layers_data[slayer]['prefix'], 'intra'))
        layer_matched_sifts_inter_dir = os.path.join(matched_sifts_dir, os.path.join(layers_data[slayer]['prefix'], 'inter'))
        create_dir(layer_matched_sifts_intra_dir)
        create_dir(layer_matched_sifts_inter_dir)

        all_layers.append(layer)

        # Skip the section if already have the optimized result
        opt_montage_json = os.path.join(args.output_dir, "{0}_montaged.json".format(tiles_fname_prefix))
        if os.path.exists(opt_montage_json):
            print "Previously optimizing (affine) layer: {0}, skipping all pre-computations".format(slayer)
            continue

        #job_sift = None
        job_multi_sift = None
        job_match = None
        job_opt_montage = None

        mfovs = set()

        # create the sift features of these tiles (per mfov)
        prev_mfov = None
        for i, ts in enumerate(cur_tilespec):
            imgurl = ts["mipmapLevels"]["0"]["imageUrl"]
            tile_fname = os.path.basename(imgurl).split('.')[0]

            cur_wafer_mfov = str(ts["mfov"]).zfill(6)
            mfov_sifts_dir = os.path.join(layer_sifts_dir, cur_wafer_mfov)
            create_dir(mfov_sifts_dir)

            mfovs.add(ts["mfov"])

            if ts["mfov"] != prev_mfov: # Assumes that the tiles are sorted by their mfov#
                # found new mfov, create a new multiple sift computation job
                job_multi_sift = CreateMultipleSiftFeatures(f, mfov_sifts_dir, cur_wafer_mfov, conf_fname=args.conf_file_name, threads_num=1)
                prev_mfov = ts["mfov"]
            
            # create the sift features of these tiles
            sifts_json = os.path.join(mfov_sifts_dir, "{0}_sifts_{1}.h5py".format(tiles_fname_prefix, tile_fname))
            if not os.path.exists(sifts_json):
                print "Computing tile  sifts: {0}".format(tile_fname)
                job_multi_sift.add_job(sifts_json, i)
                #job_sift = CreateSiftFeatures(f, sifts_json, i, conf_fname=args.conf_file_name, threads_num=2)
                jobs[slayer]['sifts'][imgurl] = job_multi_sift
            layers_data[slayer]['sifts'][imgurl] = sifts_json

        # create the intra matched sifts directories
        for mfov in mfovs:
            mfov_intra_dir = os.path.join(layer_matched_sifts_intra_dir, str(mfov))
            create_dir(mfov_intra_dir)

        # A map between layer to a list of multiple matches 
        multiple_match_jobs = {}
        # read every pair of overlapping tiles, and match their sift features
        jobs_match_intra_mfovs = {}
        jobs_match_inter_mfovs = []
        indices = []
        for pair in itertools.combinations(xrange(len(cur_tilespec)), 2):
            idx1 = pair[0]
            idx2 = pair[1]
            ts1 = cur_tilespec[idx1]
            ts2 = cur_tilespec[idx2]
            # if the two tiles intersect, match them
            bbox1 = BoundingBox.fromList(ts1["bbox"])
            bbox2 = BoundingBox.fromList(ts2["bbox"])
            if bbox1.overlap(bbox2):
                imageUrl1 = ts1["mipmapLevels"]["0"]["imageUrl"]
                imageUrl2 = ts2["mipmapLevels"]["0"]["imageUrl"]
                tile_fname1 = os.path.basename(imageUrl1).split('.')[0]
                tile_fname2 = os.path.basename(imageUrl2).split('.')[0]
                index_pair = ["{}_{}".format(ts1["mfov"], ts1["tile_index"]), "{}_{}".format(ts2["mfov"], ts2["tile_index"])]
                if ts1["mfov"] == ts2["mfov"]:
                    # Intra mfov job
                    cur_match_dir = os.path.join(layer_matched_sifts_intra_dir, str(ts1["mfov"]))
                else:
                    # Inter mfov job
                    cur_match_dir = layer_matched_sifts_inter_dir
                match_json = os.path.join(cur_match_dir, "{0}_sift_matches_{1}_{2}.json".format(tiles_fname_prefix, tile_fname1, tile_fname2))
                # match the features of overlapping tiles
                if not os.path.exists(match_json):
                    print "Matching sift of tiles: {0} and {1}".format(imageUrl1, imageUrl2)
                    dependencies = [ ]
                    if imageUrl1 in jobs[slayer]['sifts'].keys():
                        if jobs[slayer]['sifts'][imageUrl1] not in dependencies: # needed because of multiple-sift job
                            dependencies.append(jobs[slayer]['sifts'][imageUrl1])
                    if imageUrl2 in jobs[slayer]['sifts'].keys():
                        if jobs[slayer]['sifts'][imageUrl2] not in dependencies: # needed because of multiple-sift job
                            dependencies.append(jobs[slayer]['sifts'][imageUrl2])

                    # Check if the job already exists
                    if ts1["mfov"] == ts2["mfov"]:
                        # Intra mfov job
                        if ts1["mfov"] in jobs[slayer]['matched_sifts']['intra'].keys():
                            job_match = jobs[slayer]['matched_sifts']['intra'][ts1["mfov"]]
                        else:
                            job_match = MatchMultipleSiftFeaturesAndFilter(cur_match_dir, layers_data[slayer]['ts'],
                                    "intra_l{}_{}".format(slayer,ts1["mfov"]),
                                    threads_num=4, wait_time=30, conf_fname=args.conf_file_name)
                            jobs[slayer]['matched_sifts']['intra'][ts1["mfov"]] = job_match
                    else:
                        # Inter mfov job
                        if jobs[slayer]['matched_sifts']['inter'] is None:
                            job_match = MatchMultipleSiftFeaturesAndFilter(cur_match_dir, layers_data[slayer]['ts'],
                                    "inter_{}".format(slayer),
                                    threads_num=4, wait_time=30, conf_fname=args.conf_file_name)
                            jobs[slayer]['matched_sifts']['inter'] = job_match
                        else:
                            job_match = jobs[slayer]['matched_sifts']['inter']
                    job_match.add_job(dependencies, layers_data[slayer]['sifts'][imageUrl1], layers_data[slayer]['sifts'][imageUrl2],
                            match_json, index_pair)


                    #jobs[slayer]['matched_sifts'].append(job_match)
                layers_data[slayer]['matched_sifts'].append(match_json)

        # Create a single file that lists all tilespecs and a single file that lists all pmcc matches (the os doesn't support a very long list)
        matches_list_file = os.path.join(args.workspace_dir, "{}_matched_sifts_files.txt".format(tiles_fname_prefix))
        write_list_to_file(matches_list_file, layers_data[slayer]['matched_sifts'])


        # optimize (affine) the 2d layer matches (affine)
        opt_montage_json = os.path.join(args.output_dir, "{0}_montaged.json".format(tiles_fname_prefix))
        if not os.path.exists(opt_montage_json):
            print "Optimizing (affine) layer matches: {0}".format(slayer)
            dependencies = [ ]
            if len(jobs[slayer]['sifts']) > 0:
                dependencies.extend(jobs[slayer]['sifts'].values())
            if jobs[slayer]['matched_sifts']['inter'] is not None:
                dependencies.append(jobs[slayer]['matched_sifts']['inter'])
            if jobs[slayer]['matched_sifts']['intra'] is not None and len(jobs[slayer]['matched_sifts']['intra']) > 0:
                dependencies.extend(jobs[slayer]['matched_sifts']['intra'].values())
            job_opt_montage = OptimizeMontageTransform(dependencies, layers_data[slayer]['ts'],
                matches_list_file, opt_montage_json,
                conf_fname=args.conf_file_name)
        layers_data[slayer]['optimized_montage'] = opt_montage_json

        if args.multicore_keeprunning:
            # Bundle jobs for multicore nodes, and run the ones that are ready at the moment
            Job.multicore_keep_running(run_partial=True)
 




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
        # Run all jobs locally
        Job.run_all()

