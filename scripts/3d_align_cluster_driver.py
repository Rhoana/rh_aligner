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
from update_bbox import update_bbox, read_bbox
from utils import path2url

LOGS_DIR = './logs'
MEASURE_PERFORMANCE = False

class Job(object):
    all_jobs = []

    def __init__(self):
        self.name = self.__class__.__name__ + str(len(Job.all_jobs)) + '_' + datetime.datetime.now().isoformat()
        self.output = []
        self.already_done = False
        self.processors = 1
        self.time = 60
        self.memory = 1000
        self.threads = 1
        Job.all_jobs.append(self)

    def get_done(self):
        if self.already_done:
            return True
        all_outputs = self.output if isinstance(self.output, (list, tuple)) else [self.output]
        if all([os.path.exists(f) for f in all_outputs]):
            self.already_done = True
            return True
        return False

    def dependendencies_done(self):
        for d in self.dependencies:
            if not d.get_done():
                return False
        return True

    def run(self):
        # Make sure output directories exist
        out = self.output
        if isinstance(out, basestring):
            out = [out]
        for f in out:
            if not os.path.isdir(os.path.dirname(f)):
                os.mkdir(os.path.dirname(f))
        if self.get_done():
           return 0
        print "RUN", self.name
        print " ".join(self.command())

        work_queue = "serial_requeue"
        if self.threads > 1:
            work_queue = "general"

        command_list = ["sbatch",
            "-J", self.name,                   # Job name
            "-p", work_queue,            # Work queue (partition) = general / unrestricted / interactive / serial_requeue
            "--requeue",
            #"--exclude=holy2b05105,hp1301,hp0403",           # Exclude some bad nodes - holy2b05105 did not have scratch2 mapped.
            "--ntasks", str(self.processors),        # Number of processes
            "--cpus-per-task", str(self.threads),        # Number of threads
            "-t", str(self.time),              # Time in munites 1440 = 24 hours
            "--mem-per-cpu", str(self.memory), # Max memory in MB (strict - attempts to allocate more memory will fail)
            "--open-mode=append",              # Append to log files
            "-o", "logs/out." + self.name,     # Standard out file
            "-e", "logs/error." + self.name]   # Error out file

        if len(self.dependencies) > 0:
            #print command_list
            #print self.dependency_strings()
            command_list = command_list + self.dependency_strings()

        print command_list

        process = subprocess.Popen(command_list,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if MEASURE_PERFORMANCE:
            sbatch_out, sbatch_err = process.communicate("#!/bin/bash\nperf stat -o logs/perf.{0} {1}".format(self.name, " ".join(self.command())))
        else:
            sbatch_out, sbatch_err = process.communicate("#!/bin/bash\n{0}".format(" ".join(self.command())))

        if len(sbatch_err) == 0:
            self.jobid = sbatch_out.split()[3]
            #print 'jobid={0}'.format(self.jobid)

        return 1

    def dependency_strings(self):
        dependency_string = ":".join(d.jobid for d in self.dependencies if not d.get_done())
        if len(dependency_string) > 0:
            return ["-d", "afterok:" + dependency_string]
        return []

    @classmethod
    def run_all(cls):
        for j in cls.all_jobs:
            j.run()

    @classmethod
    def keep_running(cls):
        all_jobs_complete = False
        while not all_jobs_complete:

            all_job_names = {}
            # Generate dictionary of jobs
            for j in cls.all_jobs:
                all_job_names[j.name] = True

            # Find running jobs
            sacct_output = subprocess.check_output(['sacct', '-n', '-o', 'JobID,JobName%100,State,NodeList'])

            pending_running_complete_jobs = {}
            pending = 0
            running = 0
            complete = 0
            failed = 0
            cancelled = 0
            timeout = 0
            other_status = 0
            non_matching = 0

            for job_line in sacct_output.split('\n'):

                job_split = job_line.split()
                if len(job_split) == 0:
                    continue

                job_id = job_split[0]
                job_name = job_split[1]
                job_status = job_split[2]
                node = job_split[3]
                
                if job_name in all_job_names:
                    if job_status in ['PENDING', 'RUNNING', 'COMPLETED']:
                        if job_name in pending_running_complete_jobs:
                            print 'Found duplicate job: ' + job_name
                        else:
                            pending_running_complete_jobs[job_name] = True
                            if job_status == 'PENDING':
                                pending += 1
                            elif job_status == 'RUNNING':
                                running += 1
                            elif job_status == 'COMPLETED':
                                complete += 1
                    elif job_status in ['FAILED', 'NODE_FAIL']:
                        failed += 1
                    elif job_status in ['CANCELLED', 'CANCELLED+']:
                        cancelled += 1
                    elif job_status in ['TIMEOUT']:
                        timeout += 1
                    else:
                        print "Unexpected status: {0}".format(job_status)
                        other_status += 1
                elif job_name not in ['batch', 'true', 'prolog']:
                    non_matching += 1

            run_count = 0
            for j in cls.all_jobs:
                if j.name not in pending_running_complete_jobs and j.dependendencies_done():
                    run_count += j.run()

            print 'Found {0} pending, {1} running, {2} complete, {3} failed, {4} cancelled, {5} timeout, {6} unknown status and {7} non-matching jobs.'.format(
                pending, running, complete, failed, cancelled, timeout, other_status, non_matching)

            print "Queued {0} job{1}.".format(run_count, '' if run_count == 1 else 's')

            if pending > 0 or running > 0 or run_count > 0:
                time.sleep(60)
            else:
                all_jobs_complete = True


class CreateLayerSiftFeatures(Job):
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
        self.memory = 12000
        self.time = 50
        self.output = output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'create_layer_sift_features.py'),
                self.output_file, self.jar_file, self.threads_str, self.conf_fname, self.tiles_fname]

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
        self.memory = 2000
        self.time = 30
        self.output = corr_output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python',
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
        self.memory = 2000
        self.time = 30
        self.output = output_fname
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'filter_ransac.py'),
                self.jar_file, self.conf_fname, self.output_file, self.corr_fname, self.tiles_fname]

                ##job_ransac = FilterRansac(dependencies, layers_data[i]['ts'], layers_data[i]['matched_sifts'][i + j], ransac_fname, \
                ##    args.jar_file, conf_fname=args.conf_file_name)
                ###filter_ransac(match_json, path2url(layer_to_ts_json[i]), ransac_fname, args.jar_file, conf)


class MatchLayersByMaxPMCC(Job):
    def __init__(self, dependencies, tiles_fname1, tiles_fname2, ransac_fname, image_width, image_height, fixed_layers, pmcc_output_file, jar_file, conf_fname=None, threads_num=1):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname1 = '"{0}"'.format(tiles_fname1)
        self.tiles_fname2 = '"{0}"'.format(tiles_fname2)
        self.ransac_fname = '"{0}"'.format(ransac_fname)
        self.image_width = '-W {0}'.format(image_width)
        self.image_height = '-H {0}'.format(image_height)
        self.output_file = '-o "{0}"'.format(pmcc_output_file)
        self.jar_file = '-j "{0}"'.format(jar_file)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        if fixed_layers is None:
            self.fixed_layers = ''
        else:
            self.fixed_layers = '-f {0}'.format(" ".join(str(f) for f in fixed_layers))
        self.threads = threads_num
        self.threads_str = '-t {0}'.format(threads_num)
        self.dependencies = dependencies
        self.memory = 8000
        self.time = 30
        self.output = pmcc_output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'match_layers_by_max_pmcc.py'),
                self.output_file, self.fixed_layers, self.jar_file, self.conf_fname, self.threads_str, self.image_width, self.image_height,
                self.tiles_fname1, self.tiles_fname2, self.ransac_fname]


                ##job_pmcc = MatchLayersByMaxPMCC(dependencies, layers_data[i]['ts'], layers_data[i + j]['ts'], \
                ##    layers_data[i]['ransac'][i + j], imageWidth, imageHeight, \
                ##    [ fixed_layer ], pmcc_fname, args.jar_file, conf_fname=args.conf_file_name)
                ###match_layers_by_max_pmcc(args.jar_file, layer_to_ts_json[i], layer_to_ts_json[i + j], ransac_fname, imageWidth, imageHeight, [fixed_layer], pmcc_fname, conf)


class OptimizeLayersElastic(Job):
    def __init__(self, dependencies, tiles_fnames, corr_fnames, image_width, image_height, fixed_layers, output_dir, jar_file, conf_fname=None):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fnames = '--tile_files {0}'.format(" ".join(tiles_fnames))
        self.corr_fnames = '--corr_files {0}'.format(" ".join(corr_fnames))
        self.output_dir = '-o "{0}"'.format(output_dir)
        self.image_width = '-W {0}'.format(image_width)
        self.image_height = '-H {0}'.format(image_height)
        if fixed_layers is None:
            self.fixed_layers = ''
        else:
            self.fixed_layers = '-f {0}'.format(" ".join(str(f) for f in fixed_layers))
        self.jar_file = '-j "{0}"'.format(jar_file)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        self.dependencies = dependencies
        self.memory = 12000
        self.time = 30
        self.output = output_dir
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'optimize_layers_elastic.py'),
                self.output_dir, self.jar_file, self.conf_fname, self.image_width, self.image_height, self.fixed_layers, self.tiles_fnames, self.corr_fnames]



    ##job_optimize = OptimizeLayersElastic(dependencies, all_ts_files, all_pmcc_files, \
    ##    imageWidth, imageHeight, [ fixed_layer ], args.output_dir, args.jar_file, conf_fname=args.conf_file_name)
    ###optimize_layers_elastic(all_ts_files, all_pmcc_files, imageWidth, imageHeight, [fixed_layer], args.output_dir, args.jar_file, conf)




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


def create_dir(path):
    # create a directory if not found
    if not os.path.exists(path):
        os.makedirs(path)


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
    parser.add_argument('-k', '--keeprunning', action='store_true', 
                        help='Run all jobs and report cluster jobs execution stats')

    args = parser.parse_args() 

    assert 'ALIGNER' in os.environ
    #assert 'VIRTUAL_ENV' in os.environ


    # create a workspace directory if not found
    create_dir(args.workspace_dir)

    # Make sure the logs directory exists
    if not os.path.exists(LOGS_DIR) or not os.path.isdir(os.path.dirname(LOGS_DIR)):
        os.makedirs(LOGS_DIR)

    sifts_dir = os.path.join(args.workspace_dir, "sifts")
    create_dir(sifts_dir)
    matched_sifts_dir = os.path.join(args.workspace_dir, "matched_sifts")
    create_dir(matched_sifts_dir)
    after_ransac_dir = os.path.join(args.workspace_dir, "after_ransac")
    create_dir(after_ransac_dir)
    matched_pmcc_dir = os.path.join(args.workspace_dir, "matched_pmcc")
    create_dir(matched_pmcc_dir)



    # Get all input json files (one per section) into a dictionary {json_fname -> [filtered json fname, sift features file, etc.]}
    json_files = dict((jf, {}) for jf in (glob.glob(os.path.join(args.tiles_dir, '*.json'))))





    all_layers = []
    all_ts_files = []
    jobs = {}
    layers_data = {}
    #layer_to_sifts = {}
    #layer_to_ts_json = {}
    #layer_to_json_prefix = {}

    # Find all images width and height (for the mesh)
    imageWidth = None
    imageHeight = None

    for f in json_files.keys():
        tiles_fname_prefix = os.path.splitext(os.path.basename(f))[0]

        # read the layer from the file
        layer = read_layer_from_file(f)

        slayer = str(layer)

        if not (slayer in layers_data.keys()):
            layers_data[slayer] = {}
            jobs[slayer] = {}
            jobs[slayer]['sifts'] = []


        all_layers.append(layer)

        # update the bbox of each section
        #after_bbox_json = os.path.join(after_bbox_dir, "{0}{1}.json".format(tiles_fname_prefix, bbox_suffix)) 
        #if not os.path.exists(after_bbox_json):
        #    print "Updating bounding box of {0}".format(tiles_fname_prefix)
        #    update_bbox(args.jar_file, tiles_fname, out_dir=after_bbox_dir, out_suffix=bbox_suffix)
        #bbox = read_bbox(after_bbox_json)
        bbox = read_bbox(f)
        if imageWidth is None or imageWidth < (bbox[1] - bbox[0]):
            imageWidth = bbox[1] - bbox[0]
        if imageHeight is None or imageHeight < bbox[3] - bbox[2]:
            imageHeight = bbox[3] - bbox[2]

        # create the sift features of these tiles
        sifts_json = os.path.join(sifts_dir, "{0}_sifts.json".format(tiles_fname_prefix))
        if not os.path.exists(sifts_json):
            sift_job = CreateLayerSiftFeatures(f, sifts_json, args.jar_file, conf_fname=args.conf_file_name, threads_num=4)
            jobs[slayer]['sifts'].append(sift_job)


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
            print "Error missing layers between: {1} and {2}".format(all_layers[i], all_layers[i + 1])
            sys.exit(1)

    print "Found the following layers: {0}".format(all_layers)

    # Set the first layer as a fixed layer
    fixed_layer = all_layers[0]


    # Match and optimize each two layers in the required distance
    all_pmcc_files = []
    pmcc_jobs = []
    for i in all_layers:
        si = str(i)
        layers_data[si]['matched_sifts'] = {}
        layers_data[si]['ransac'] = {}
        layers_data[si]['matched_pmcc'] = {}
        layers_to_process = min(i + args.max_layer_distance, all_layers[-1] + 1) - i
        for j in range(1, layers_to_process):
            sij = str(i + j)
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
            layers_data[si]['matched_sifts'][sij] = match_json


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
            layers_data[si]['ransac'][sij] = ransac_fname

            # match by max PMCC the two layers
            pmcc_fname = os.path.join(matched_pmcc_dir, "{0}_{1}_match_pmcc.json".format(fname1_prefix, fname2_prefix))
            if not os.path.exists(pmcc_fname):
                print "Matching layers by Max PMCC: {0} and {1}".format(i, i + j)
                dependencies = [ ]
                if job_ransac != None:
                    dependencies.append(job_ransac)
                if job_match != None:
                    dependencies.append(job_match)
                for dep in jobs[si]['sifts']:
                    dependencies.append(dep)
                for dep in jobs[sij]['sifts']:
                    dependencies.append(dep)

                job_pmcc = MatchLayersByMaxPMCC(dependencies, layers_data[si]['ts'], layers_data[sij]['ts'], \
                    layers_data[si]['ransac'][sij], imageWidth, imageHeight, \
                    [ fixed_layer ], pmcc_fname, args.jar_file, conf_fname=args.conf_file_name, threads_num=8)
                #match_layers_by_max_pmcc(args.jar_file, layer_to_ts_json[i], layer_to_ts_json[i + j], ransac_fname, imageWidth, imageHeight, [fixed_layer], pmcc_fname, conf)
                pmcc_jobs.append(job_pmcc)
            layers_data[si]['matched_pmcc'][sij] = pmcc_fname

            all_pmcc_files.append(pmcc_fname)





    # Optimize all layers to a single 3d image
    create_dir(args.output_dir)
    dependencies = pmcc_jobs
    job_optimize = OptimizeLayersElastic(dependencies, all_ts_files, all_pmcc_files, \
        imageWidth, imageHeight, [ fixed_layer ], args.output_dir, args.jar_file, conf_fname=args.conf_file_name)
    #optimize_layers_elastic(all_ts_files, all_pmcc_files, imageWidth, imageHeight, [fixed_layer], args.output_dir, args.jar_file, conf)




    # Run all jobs

    if args.keeprunning:
        Job.keep_running()
    else:
        Job.run_all()

