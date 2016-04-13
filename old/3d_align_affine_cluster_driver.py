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
from utils import path2url, write_list_to_file, create_dir, read_layer_from_file, parse_range
from job import Job
from bounding_box import BoundingBox


class CreateMeshes(Job):
    def __init__(self, tiles_fname, output_dir, jar_file, threads_num=1):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.output_dir = '-o "{0}"'.format(output_dir)
        self.jar_file = '-j "{0}"'.format(jar_file)
        self.dependencies = [ ]
        self.threads = threads_num
        self.threads_str = "-t {0}".format(threads_num)
        self.memory = 64000
        self.time = 200
        self.is_java_job = True
        self.output = output_dir
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'create_meshes.py'),
                self.output_dir, self.jar_file, self.threads_str, self.tiles_fname]


class CreateLayerSiftFeatures(Job):
    def __init__(self, dependencies, tiles_fname, output_file, jar_file, meshes_dir=None, conf_fname=None, threads_num=1):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.output_file = '-o "{0}"'.format(output_file)
        self.jar_file = '-j "{0}"'.format(jar_file)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        if meshes_dir is None:
            self.meshes_dir = ''
        else:
            self.meshes_dir = '--meshes_dir "{0}"'.format(meshes_dir)
        self.dependencies = dependencies
        self.threads = threads_num
        self.threads_str = "-t {0}".format(threads_num)
        self.memory = 13000
        self.time = 400
        self.is_java_job = True
        self.output = output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'create_layer_sift_features.py'),
                self.output_file, self.jar_file, self.threads_str, self.meshes_dir, self.conf_fname, self.tiles_fname]

class MatchLayersSiftFeatures(Job):
    def __init__(self, dependencies, tiles_fname1, features_fname1, tiles_fname2, features_fname2, corr_output_file, jar_file, conf_fname=None, threads_num=1):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname1 = '"{0}"'.format(tiles_fname1)
        self.features_fname1 = '"{0}"'.format(features_fname1)
        self.tiles_fname2 = '"{0}"'.format(tiles_fname2)
        self.features_fname2 = '"{0}"'.format(features_fname2)
        self.output_file = '-o "{0}"'.format(corr_output_file)
        self.jar_file = '-j "{0}"'.format(jar_file)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        self.dependencies = dependencies
        self.threads = threads_num
        self.threads_str = "-t {0}".format(threads_num)
        self.memory = 5000
        self.time = 30
        self.is_java_job = True
        self.output = corr_output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'match_layers_sift_features.py'),
                self.output_file, self.jar_file, self.conf_fname, self.tiles_fname1, self.features_fname1, self.tiles_fname2, self.features_fname2]


                ##job_match = MatchLayersSiftFeatures(dependencies, layers_data[i]['ts'], layers_data[i]['sifts'], \
                ##    layers_data[i + j]['ts'], layers_data[i + j]['sifts'], match_json, \
                ##    args.jar_file, conf_fname=args.conf_file_name)


class FilterRansac(Job):
    def __init__(self, dependencies, tiles_fname, corr_fname, output_fname, jar_file, conf_fname=None):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.corr_fname = '"{0}"'.format(corr_fname)
        self.output_file = '-o "{0}"'.format(output_fname)
        self.jar_file = '-j "{0}"'.format(jar_file)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        self.dependencies = dependencies
        self.memory = 5000
        self.time = 30
        self.is_java_job = True
        self.output = output_fname
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'filter_ransac.py'),
                self.jar_file, self.conf_fname, self.output_file, self.corr_fname, self.tiles_fname]

                ##job_ransac = FilterRansac(dependencies, layers_data[i]['ts'], layers_data[i]['matched_sifts'][i + j], ransac_fname, \
                ##    args.jar_file, conf_fname=args.conf_file_name)
                ###filter_ransac(match_json, path2url(layer_to_ts_json[i]), ransac_fname, args.jar_file, conf)


class OptimizeLayersAffine(Job):
    def __init__(self, dependencies, outputs, tiles_fnames, corr_fnames, model_fnames, fixed_layers, output_dir, max_layer_distance, jar_file, conf_fname=None, skip_layers=None, threads_num=1, manual_matches=None):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fnames = '--tile_files {0}'.format(" ".join(tiles_fnames))
        self.corr_fnames = '--corr_files {0}'.format(" ".join(corr_fnames))
        self.model_fnames = '--model_files {0}'.format(" ".join(model_fnames))
        self.output_dir = '-o "{0}"'.format(output_dir)
        if fixed_layers is None:
            self.fixed_layers = ''
        else:
            self.fixed_layers = '-f {0}'.format(" ".join(str(f) for f in fixed_layers))
        self.jar_file = '-j "{0}"'.format(jar_file)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        if skip_layers is None:
            self.skip_layers = ''
        else:
            self.skip_layers = '-s "{0}"'.format(skip_layers)
        if manual_matches is None:
            self.manual_matches = ''
        else:
            self.manual_matches = '-M {0}'.format(" ".join(manual_matches))
        self.max_layer_distance = '-d {}'.format(max_layer_distance)
        self.threads = threads_num
        self.threads_str = '-t {0}'.format(threads_num)
        self.dependencies = dependencies
        self.memory = 50000
        self.time = 1400
        self.is_java_job = True
        self.output = outputs
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python -u',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'optimize_layers_affine.py'),
                self.output_dir, self.jar_file, self.conf_fname, self.fixed_layers,
                self.max_layer_distance, self.threads_str, self.manual_matches, self.skip_layers, self.tiles_fnames, self.corr_fnames, self.model_fnames]



    ##job_optimize = OptimizeLayersElastic(dependencies, all_ts_files, all_pmcc_files, \
    ##    imageWidth, imageHeight, [ fixed_layer ], args.output_dir, args.jar_file, conf_fname=args.conf_file_name)
    ###optimize_layers_elastic(all_ts_files, all_pmcc_files, imageWidth, imageHeight, [fixed_layer], args.output_dir, args.jar_file, conf)





###############################
# Driver
###############################
if __name__ == '__main__':



    # Command line parser
    parser = argparse.ArgumentParser(description='Aligns a given set of images using the SLURM cluster commands.')
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
    parser.add_argument('-d', '--max_layer_distance', type=int, 
                        help='the largest distance between two layers to be matched (default: 1)',
                        default=1)
    parser.add_argument('--auto_add_model', action="store_true", 
                        help='automatically add the identity model, if a model is not found')
    parser.add_argument('-s', '--skip_layers', type=str, 
                        help='the range of layers (sections) that will not be processed e.g., "2,3,9-11,18" (default: no skipped sections)',
                        default=None)
    # parser.add_argument('-M', '--manual_match', type=str, nargs="*",
    #                     help='pairs of layers (sections) that will need to be manually aligned (not part of the max_layer_distance) e.g., "2:10,7:21" (default: none)',
    #                     default=None)
    parser.add_argument('-F', '--fix_every_nth', type=int, 
                        help='each Nth layer will be fixed (default: only middle layer)',
                        default=0)
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
    after_ransac_dir = os.path.join(args.workspace_dir, "after_ransac")
    create_dir(after_ransac_dir)



    # Get all input json files (one per section) into a dictionary {json_fname -> [filtered json fname, sift features file, etc.]}
    json_files = dict((jf, {}) for jf in (glob.glob(os.path.join(args.tiles_dir, '*.json'))))



    skipped_layers = parse_range(args.skip_layers)


    all_layers = []
    all_ts_files = []
    jobs = {}
    layers_data = {}
    #layer_to_sifts = {}
    #layer_to_ts_json = {}
    #layer_to_json_prefix = {}

    all_running_jobs = []


    for f in json_files.keys():
        tiles_fname_prefix = os.path.splitext(os.path.basename(f))[0]

        # read the layer from the file
        layer = read_layer_from_file(f)

        if layer in skipped_layers:
            continue

        slayer = str(layer)

        if not (slayer in layers_data.keys()):
            layers_data[slayer] = {}
            jobs[slayer] = {}
            jobs[slayer]['sifts'] = []


        all_layers.append(layer)



        # create the sift features of these tiles
        sifts_json = os.path.join(sifts_dir, "{0}_sifts.json".format(tiles_fname_prefix))
        if not os.path.exists(sifts_json):
            sift_job = None
            dependencies = [ ]
            sift_job = CreateLayerSiftFeatures(dependencies, f, sifts_json, args.jar_file, conf_fname=args.conf_file_name, threads_num=8)
            jobs[slayer]['sifts'].append(sift_job)
            all_running_jobs.append(sift_job)


        layers_data[slayer]['ts'] = f
        layers_data[slayer]['sifts'] = sifts_json
        layers_data[slayer]['prefix'] = tiles_fname_prefix
        #layer_to_sifts[layer] = sifts_json
        #layer_to_json_prefix[layer] = tiles_fname_prefix
        ##layer_to_ts_json[layer] = after_bbox_json
        #layer_to_ts_json[layer] = f
        all_ts_files.append(f)





    # Verify that all the layers are there and that there are no holes
    all_layers.sort()
    for i in range(len(all_layers) - 1):
        if all_layers[i + 1] - all_layers[i] != 1:
            for l in range(all_layers[i] + 1, all_layers[i + 1]):
                if l not in skipped_layers:
                    print "Error missing layer {} between: {} and {}".format(l, all_layers[i], all_layers[i + 1])
                    sys.exit(1)

    print "Found the following layers: {0}".format(all_layers)

    if args.fix_every_nth == 0:
        # Set the middle layer as a fixed layer
        fixed_layers = [ all_layers[len(all_layers)//2] ]
    else:
        fixed_layers = all_layers[::args.fix_every_nth]

    # manual_matches = {}
    # if args.manual_match is not None:
    #     for match in args.manual_match:
    #         # parse the manual match string
    #         match_layers = [int(l) for l in match.split(':')]
    #         # add a manual match between the lower layer and the higher layer
    #         if min(match_layers) not in manual_matches.keys():
    #             manual_matches[min(match_layers)] = []
    #         manual_matches[min(match_layers)].append(max(match_layers))


    # Match and optimize each two layers in the required distance
    all_matched_sifts_files = []
    all_model_files = []
    match_sifts_jobs = []
    filter_ransac_jobs = []
    for ei, i in enumerate(all_layers):
        si = str(i)
        layers_data[si]['matched_sifts'] = {}
        layers_data[si]['ransac'] = {}
        # layers_to_process = min(i + args.max_layer_distance + 1, all_layers[-1] + 1) - i
        # to_range = range(1, layers_to_process)
        # # add manual matches
        # if i in manual_matches.keys():
        #     for second_layer in manual_matches[i]:
        #         diff_layers = second_layer - i
        #         if diff_layers not in to_range:
        #             to_range.append(diff_layers)
        # Process all matched layers
        matched_after_layers = 0
        j = 1
        while matched_after_layers < args.max_layer_distance:
            if ei + j >= len(all_layers):
                break

            sij = str(i + j)
            if i in skipped_layers or (i+j) in skipped_layers:
                print "Skipping matching of layers {} and {}, because at least one of them should be skipped".format(i, i+j)
                j += 1
                continue

            fname1_prefix = layers_data[si]['prefix']
            fname2_prefix = layers_data[sij]['prefix']

            job_match = None
            job_ransac = None
            job_pmcc = None

            # match the features of neighboring tiles
            match_json = os.path.join(matched_sifts_dir, "{0}_{1}_sift_matches.json".format(fname1_prefix, fname2_prefix))
            if not os.path.exists(match_json):
                print "Matching layers' sifts: {0} and {1}".format(i, i + j)
                dependencies = [ ]
                for dep in jobs[si]['sifts']:
                    dependencies.append(dep)
                for dep in jobs[sij]['sifts']:
                    dependencies.append(dep)

                tiles_fname_prefix = os.path.splitext(os.path.basename(f))[0]
                job_match = MatchLayersSiftFeatures(dependencies, layers_data[si]['ts'], layers_data[si]['sifts'], \
                    layers_data[sij]['ts'], layers_data[sij]['sifts'], match_json, \
                    args.jar_file, conf_fname=args.conf_file_name)
                all_running_jobs.append(job_match)
                match_sifts_jobs.append(job_match)
            layers_data[si]['matched_sifts'][sij] = match_json
            all_matched_sifts_files.append(match_json)

            # filter and ransac the matched points
            ransac_fname = os.path.join(after_ransac_dir, "{0}_{1}_filter_ransac.json".format(fname1_prefix, fname2_prefix))
            if not os.path.exists(ransac_fname):
                print "Filter-and-Ransac of layers: {0} and {1}".format(i, i + j)
                dependencies = [ ]
                if job_match != None:
                    dependencies.append(job_match)
                for dep in jobs[si]['sifts']:
                    dependencies.append(dep)
                for dep in jobs[sij]['sifts']:
                    dependencies.append(dep)

                job_ransac = FilterRansac(dependencies, path2url(layers_data[si]['ts']), layers_data[si]['matched_sifts'][sij], ransac_fname, \
                    args.jar_file, conf_fname=args.conf_file_name)
                #filter_ransac(match_json, path2url(layer_to_ts_json[i]), ransac_fname, args.jar_file, conf)
                all_running_jobs.append(job_ransac)
                filter_ransac_jobs.append(job_ransac)
            layers_data[si]['ransac'][sij] = ransac_fname
            all_model_files.append(ransac_fname)


            j += 1
            matched_after_layers += 1



    # Create a single file that lists all tilespecs and a single file that lists all pmcc matches (the os doesn't support a very long list)
    ts_list_file = os.path.join(args.workspace_dir, "all_ts_files.txt")
    write_list_to_file(ts_list_file, all_ts_files)
    matched_sifts_list_file = os.path.join(args.workspace_dir, "all_matched_sifts_files.txt")
    write_list_to_file(matched_sifts_list_file, all_matched_sifts_files)
    model_list_file = os.path.join(args.workspace_dir, "all_model_files.txt")
    write_list_to_file(model_list_file, all_model_files)


    # Optimize all layers to a single 3d image
    create_dir(args.output_dir)
    sections_outputs = []
    for i in all_layers:
        out_section = os.path.join(args.output_dir, os.path.basename(layers_data[str(i)]['ts']))
        sections_outputs.append(out_section)

    dependencies = all_running_jobs
    job_optimize = OptimizeLayersAffine(dependencies, sections_outputs, [ ts_list_file ], [ matched_sifts_list_file ], [ model_list_file ],
        fixed_layers, args.output_dir, args.max_layer_distance, args.jar_file, conf_fname=args.conf_file_name,
        skip_layers=args.skip_layers, threads_num=8)


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

