#
# Executes the alignment process jobs on the cluster (based on Rhoana's driver).
# It takes a collection of tilespec files of different layers and outputs tilespec files for
# 2d and 3d alignment of these tiles.
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
            "-n", str(self.processors),        # Number of processors
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


class FilterTiles(Job):
    def __init__(self, tiles_fname, output_file, bounding_box=None):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.output_file = '-o "{0}"'.format(output_file)
        if bounding_box is None:
            self.bounding_box = ''
        else:
            self.bounding_box = '-b "{0}"'.format(bounding_box)
        self.dependencies = []
        self.memory = 200
        self.time = 20
        self.output = output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'filter_tiles.py'),
                self.output_file, self.bounding_box, self.tiles_fname]

class CreateSiftFeatures(Job):
    def __init__(self, dependencies, tiles_fname, output_file, jar_file, conf_fname=None, threads_num=1):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.output_file = '-o "{0}"'.format(output_file)
        self.jar_file = '-j "{0}"'.format(jar_file)
        if conf_fname is None:
            self.conf_fname = ''
        else:
            self.conf_fname = '-c "{0}"'.format(conf_fname)
        self.threads = threads_num
        self.threads_num = "-t {0}".format(threads_num)
        self.dependencies = dependencies
        self.memory = 8000
        self.time = 50
        self.output = output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'create_sift_features.py'),
                self.output_file, self.jar_file, self.threads_num, self.conf_fname, self.tiles_fname]

class MatchSiftFeatures(Job):
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
        self.memory = 2000
        self.time = 30
        self.output = corr_output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'match_sift_features.py'),
                self.output_file, self.jar_file, self.conf_fname, self.tiles_fname, self.features_fname]

class OptimizeMontageTransform(Job):
    def __init__(self, dependencies, tiles_fname, corr_fname, output_fname, jar_file, conf_fname=None):
        Job.__init__(self)
        self.already_done = False
        self.tiles_fname = '"{0}"'.format(tiles_fname)
        self.corr_fname = '"{0}"'.format(corr_fname)
        self.output_file = '"{0}"'.format(output_fname)
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
                os.path.join(os.environ['ALIGNER'], 'scripts', 'optimize_montage_transform.py'),
                self.jar_file, self.conf_fname, self.corr_fname, self.tiles_fname, self.output_file]


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
    parser.add_argument('-o', '--output_file_name', type=str, 
                        help='the file that includes the output to be rendered in json format (default: output.json)',
                        default='output.json')
    parser.add_argument('-j', '--jar_file', type=str, 
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')
    # the default bounding box is as big as the image can be
    parser.add_argument('-b', '--bounding_box', type=str, 
                        help='the bounding box of the part of image that needs to be aligned format: "from_x to_x from_y to_y" (default: all tiles)',
                        default='{0} {1} {2} {3}'.format((-sys.maxint - 1), sys.maxint, (-sys.maxint - 1), sys.maxint))
    parser.add_argument('-c', '--conf_file_name', type=str, 
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)
    parser.add_argument('-k', '--keeprunning', action='store_true', 
                        help='Run all jobs and report cluster jobs execution stats')

    args = parser.parse_args() 

    assert 'ALIGNER' in os.environ
    #assert 'VIRTUAL_ENV' in os.environ


    # create a workspace directory if not found
    if not os.path.exists(args.workspace_dir):
        os.makedirs(args.workspace_dir)

    # Make sure the logs directory exists
    if not os.path.exists(LOGS_DIR) or not os.path.isdir(os.path.dirname(LOGS_DIR)):
        os.makedirs(LOGS_DIR)


    # Get all input json files (one per section) into a dictionary {json_fname -> [filtered json fname, sift features file, etc.]}
    json_files = dict((jf, {}) for jf in (glob.glob(os.path.join(args.tiles_dir, '*.json'))))


    # Filter the tiles according to the bounding box
    filtered_jobs = {}
    for f in json_files.keys():
        tiles_fname_prefix = os.path.splitext(os.path.basename(f))[0]
        filter_json = os.path.join(args.workspace_dir, "{0}_filterd.json".format(tiles_fname_prefix))
        json_files[f]['filtered'] = filter_json
        filtered_jobs[f] = FilterTiles(f, filter_json, args.bounding_box)

    # Create sift features
    sift_jobs = {}
    for f in json_files.keys():
        dependencies = [ filtered_jobs[f] ]
        tiles_fname_prefix = os.path.splitext(os.path.basename(f))[0]
        sifts_json = os.path.join(args.workspace_dir, "{0}_sifts.json".format(tiles_fname_prefix))
        json_files[f]['sifts'] = sifts_json
        sift_jobs[f] = CreateSiftFeatures(dependencies, json_files[f]['filtered'], sifts_json, args.jar_file, conf_fname=args.conf_file_name, threads_num=2)


    # Match sift features of adjacent tiles
    match_jobs = {}
    for f in json_files.keys():
        dependencies = [ filtered_jobs[f], sift_jobs[f] ]
        tiles_fname_prefix = os.path.splitext(os.path.basename(f))[0]
        matches_json = os.path.join(args.workspace_dir, "{0}_matches.json".format(tiles_fname_prefix))
        json_files[f]['matches'] = matches_json
        match_jobs[f] = MatchSiftFeatures(dependencies, json_files[f]['filtered'], json_files[f]['sifts'], matches_json,\
            args.jar_file, conf_fname=args.conf_file_name)

    # Optimize matches
    opt_montage_jobs = {}
    for f in json_files.keys():
        dependencies = [ filtered_jobs[f], match_jobs[f] ]
        tiles_fname_prefix = os.path.splitext(os.path.basename(f))[0]
        optimized_json = os.path.join(args.workspace_dir, "{0}_opt_montage.json".format(tiles_fname_prefix))
        json_files[f]['opt_montage'] = matches_json
        opt_montage_jobs[f] = OptimizeMontageTransform(dependencies, json_files[f]['filtered'], json_files[f]['matches'], optimized_json,\
            args.jar_file, conf_fname=args.conf_file_name)

    print json_files


    # Run all jobs

    if args.keeprunning:
        Job.keep_running()
    else:
        Job.run_all()

