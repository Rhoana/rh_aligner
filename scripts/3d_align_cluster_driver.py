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
RUN_LOCAL = False
USE_QSUB = False
USE_SBATCH = True


#Multicore settings
##MAX_CORES = 16
MAX_CPUS_PER_NODE = 60
MAX_MEMORY_MB = 128000
MIN_TIME = 600
MAX_JOBS_TO_SUBMIT = 100
TIME_FACTOR = 4


class Job(object):
    all_jobs = []

    def __init__(self):
        self.name = self.__class__.__name__ + str(len(Job.all_jobs)) + '_' + datetime.datetime.now().isoformat()
        self.jobid = None
        self.output = []
        self.already_done = False
        ##self.processors = 1
        self.time = 60
        self.memory = 1000
        self.threads = 1
        self.is_java_job = False
        Job.all_jobs.append(self)

    def get_threads_num(self):
        if self.is_java_job:
            return self.threads + 1
        return self.threads

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
        if self.get_threads_num() > 1:
            work_queue = "general"

        if RUN_LOCAL:
            subprocess.check_call(self.command())
        elif USE_SBATCH:
            command_list = ["sbatch",
                "-J", self.name,                   # Job name
                "-p", work_queue,            # Work queue (partition) = general / unrestricted / interactive / serial_requeue
                "--requeue",
                #"--exclude=holy2b05105,hp1301,hp0403",           # Exclude some bad nodes - holy2b05105 did not have scratch2 mapped.
                ##"--ntasks", str(self.processors),        # Number of processes
                "--ntasks", str(1),        # Number of processes
                "--cpus-per-task", str(self.get_threads_num()),        # Number of threads
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

    # All given job must have the same number of threads
    @classmethod
    def run_job_blocks(cls, job_block_list, required_cores, required_memory, required_full_time, required_threads):
        block_name = 'JobBlock{0}.'.format(cls.block_count) + job_block_list[0][0].name
        cls.block_count += 1
        print "RUNNING JOB BLOCK: " + block_name
        print "{0} blocks, {1} jobs, {2} cores, {3}MB memory, {4}m time, {5} threads per task.".format(
            len(job_block_list), [len(jb) for jb in job_block_list], required_cores, required_memory, required_full_time, required_threads)
        full_command = "#!/bin/bash\n"
        dependency_set = set()

        # Find all dependencies for all jobs
        for job_block in job_block_list:
            for j in job_block:
                for d in j.dependencies:
                    if not d.get_done() and d.jobid is not None:
                        if USE_SBATCH or USE_QSUB:
                            dependency_set.add(d.jobid)
                        # else:
                        #     dependency_set.add(d.name)

        work_queue = "serial_requeue"
        if required_threads > 1:
            work_queue = "general"


        if USE_SBATCH:
            command_list = ["sbatch",
                "-J", block_name,                   # Job name
                "-p", work_queue,            # Work queue (partition) = general / unrestricted / interactive / serial_requeue
                "--requeue",
                #"--exclude=holy2b05105,hp1301,hp0403",           # Exclude some bad nodes - holy2b05105 did not have scratch2 mapped.
                "--ntasks", str(required_cores),        # Number of processes
                "--cpus-per-task", str(required_threads),        # Number of threads
#                "-n", str(required_cores),        # Number of processors
                "-t", str(required_full_time),              # Time in munites 1440 = 24 hours
                "--mem-per-cpu", str(required_memory), # Max memory in MB (strict - attempts to allocate more memory will fail)
                "--open-mode=append",              # Append to log files
                "-o", "logs/out." + block_name,     # Standard out file
                "-e", "logs/error." + block_name]   # Error out file

        elif USE_QSUB:
            command_list = ["qsub"]#,
                # "-N", block_name,                   # Job name
                # "-A", 'hvd113',                    # XSEDE Allocation
                # "-q", QSUB_WORK_QUEUE,                    # Work queue (partition) = general / unrestricted / interactive / serial_requeue
                # "-l", 'nodes=1:ppn={0},walltime={1}:00'.format(str(required_cores), required_full_time),  # Number of processors
                # #"-l", 'walltime={0}:00'.format(self.time),             # Time in munites 1440 = 24 hours
                # #"-l", '-mppmem={0}'.format(self.memory),               # Max memory per cpu in MB (strict - attempts to allocate more memory will fail)
                # "-e", "logs/outerror." + block_name('_')[0],      # Error out file
                # "-j", "eo"]                                            # Join standard out file to error file

            # Better to use file input rather than command line inputs (according to XSEDE helpdesk)
            # Request MAX_CORES so that memory requirement is also met
            full_command += (
               "#PBS -N {0}\n".format(block_name) +
               "#PBS -A hvd113\n" +
               "#PBS -q {0}\n".format(QSUB_WORK_QUEUE) +
               "#PBS -l nodes=1:ppn={0}:native,walltime={1}:00\n".format(str(MAX_CORES), required_full_time) +
               "#PBS -e logs/outerror.{0}\n".format(block_name.split('_')[0]) +
               "#PBS -j eo\n")

        if len(dependency_set) > 0:
            if USE_SBATCH:
                dependency_string = ":".join(d for d in dependency_set)
                if len(dependency_string) > 0:
                    print "depends on jobs:" + dependency_string
                    command_list += ["-d", "afterok:" + dependency_string]
            elif USE_QSUB:
                dependency_string = ":".join(d for d in dependency_set)
                if len(dependency_string) > 0:
                    print "depends on jobs:" + dependency_string
                    full_command += "#PBS -W depend=afterok:" + dependency_string + "\n"
            else:
                command_list += " && ".join("done(%s)" % d for d in dependency_set)

        if USE_SBATCH:
            full_command += "date\n"
        elif USE_QSUB:
            full_command += "cd $PBS_O_WORKDIR\ndate\n"

        # Generate job block commands
        for job_block in job_block_list:
            block_commands = ''
            for j in job_block:
                block_commands += '{0} &\n'.format(' '.join(j.command()))
                print j.name
            full_command += '{0}wait\ndate\n'.format(block_commands)

        # # Test job ids
        # for job_block in job_block_list:
        #     for j in job_block:
        #         j.jobid = str(cls.block_count - 1)

        # print command_list
        # print full_command

        # Submit job
        process = subprocess.Popen(command_list, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        submit_out, submit_err = process.communicate(full_command)

        # Process output
        if len(submit_err) == 0:
            if USE_SBATCH:
                new_jobid = submit_out.split()[3]
            elif USE_QSUB:
                new_jobid = submit_out.split('.')[0]
            print 'jobid={0}'.format(new_jobid)
            for job_block in job_block_list:
                for j in job_block:
                    j.jobid = new_jobid

    @classmethod
    def multicore_run_all(cls):
        all_jobs_optimizer = JobBlockOrganizer()

        for j in cls.all_jobs:

            # Make sure output directories exist
            out = j.output
            if isinstance(out, basestring):
                out = [out]
            for f in out:
                if not os.path.isdir(os.path.dirname(f)):
                    os.mkdir(os.path.dirname(f))

            if j.get_done():
                continue

            all_jobs_optimizer.add_job(j)

        all_jobs_optimizer.run_all()

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

class JobBlock(object):
    block_count = 0
    all_job_blocks = []
    jobs_to_job_block = {}

    # Represnts a block of jobs that needs to be executed on the cluster
    def __init__(self, threads):
        self.pending = True
        self.job_block_list = []
        self.required_threads = threads
        self.required_memory = 0
        self.required_full_time = 0
        self.jobs_count = 0
        JobBlock.block_count += 1
        self.block_num = JobBlock.block_count
        JobBlock.all_job_blocks.append(self)
        self.job_block_dependencies = set()

    def can_add_job(self, job):
        # If the job was already submitted, return false
        if not self.pending:
            # print "Cannot add: not pending:\n{0} to block {1}".format(job.command, self.block_num)
            return False

        # If the job has a different number of threads, return false
        if self.required_threads != job.get_threads_num():
            # print "Cannot add: not threads:\n{0} to block {1}".format(job.command, self.block_num)
            return False

        # If the job requires more threads than available, return false
        if self.required_threads * (self.jobs_count + 1) > MAX_CPUS_PER_NODE:
            # print "Cannot add: no cpus left:\n{0} to block {1}".format(job.command, self.block_num)
            return False

        # If the job requires more memory than available, return false
        if self.required_memory + job.memory > MAX_MEMORY_MB:
            # print "Cannot add: no memory left:\n{0} to block {1}".format(job.command, self.block_num)
            return False

        # If the job has a dependent job in the same job_list, return false
        for d in job.dependencies:
            if d in self.job_block_list:
                # print "Cannot add: job is conflicted with another job:\n{0} to block {1}".format(job.command, self.block_num)
                return False

        # Check the JobBlock dependencies by copying the current dependencies,
        # adding the new ones, and computing the transitive closure
        # We then need to make sure we are not in the closure's output (otherwise, there is a cycle)
        # (optimization, we first check that the given job's dependencies are not already there)
        # for d in job.dependencies:
        #     if d in JobBlock.jobs_to_job_block.keys():
        #         other_job_block = JobBlock.jobs_to_job_block[d]
        #         if (other_job_block == self) or (other_job_block in self.job_block_dependencies):
        #             # print "Cannot add: job is conflicted with another dependency"
        #             return False
        for d in job.dependencies:
            if d in JobBlock.jobs_to_job_block.keys():
                other_job_block = JobBlock.jobs_to_job_block[d]
                if other_job_block == self:
                    # print "Cannot add: job is conflicted with another dependency:\n{0} to block {1}".format(job.command, self.block_num)
                    return False


        return True

    def add_job(self, job):
        self.job_block_list.append(job)
        self.jobs_count += 1
        self.required_memory += job.memory
        self.required_full_time = max(self.required_full_time, job.time)
        JobBlock.jobs_to_job_block[job] = self
        # Update the JobBlock dependencies
        for d in job.dependencies:
            if d in JobBlock.jobs_to_job_block.keys():
                other_job_block = JobBlock.jobs_to_job_block[d]
                self.job_block_dependencies.add(other_job_block)



    def is_pending(self):
        return self.pending

    def submit_block(self):
        # If there are no jobs, do nothing
        if self.jobs_count == 0:
            return

        # If th ejob block was already submitted, do nothing
        if not self.pending:
            return

        self.pending = False

        # receursively execute all the dependencies
        for other_job_block in self.job_block_dependencies:
            if other_job_block.is_pending():
                other_job_block.submit_block()

        block_name = 'JobBlock{0}.'.format(self.block_num) + self.job_block_list[0].name
        print "RUNNING JOB BLOCK: " + block_name
        print "{0} jobs(tasks), {1} threads per task, {2}MB memory, {3}m time.".format(
            self.jobs_count, self.required_threads, self.required_memory, self.required_full_time)

        full_command = "#!/bin/bash\n"
        dependency_set = set()

        # Find all dependencies for all jobs
        for j in self.job_block_list:
            for d in j.dependencies:
                if not d.get_done() and d.jobid is not None:
                    if USE_SBATCH or USE_QSUB:
                        dependency_set.add(d.jobid)
                    # else:
                    #     dependency_set.add(d.name)

        work_queue = "serial_requeue"
        # if self.required_threads > 1:
        #     work_queue = "general"


        if USE_SBATCH:
            command_list = ["sbatch",
                "-J", block_name,                   # Job name
                "-p", work_queue,            # Work queue (partition) = general / unrestricted / interactive / serial_requeue
                "--requeue",
                #"--exclude=holy2b05105,hp1301,hp0403",           # Exclude some bad nodes - holy2b05105 did not have scratch2 mapped.
                "--ntasks", str(self.jobs_count),        # Number of processes
                "--cpus-per-task", str(self.required_threads),        # Number of threads
#                "-n", str(required_cores),        # Number of processors
                "-t", str(self.required_full_time),              # Time in munites 1440 = 24 hours
                "--mem", str(self.required_memory), # Max memory in MB (strict - attempts to allocate more memory will fail)
                "--open-mode=append",              # Append to log files
                "-o", "logs/out." + block_name,     # Standard out file
                "-e", "logs/error." + block_name]   # Error out file

        elif USE_QSUB:
            command_list = ["qsub"]#,
                # "-N", block_name,                   # Job name
                # "-A", 'hvd113',                    # XSEDE Allocation
                # "-q", QSUB_WORK_QUEUE,                    # Work queue (partition) = general / unrestricted / interactive / serial_requeue
                # "-l", 'nodes=1:ppn={0},walltime={1}:00'.format(str(required_cores), required_full_time),  # Number of processors
                # #"-l", 'walltime={0}:00'.format(self.time),             # Time in munites 1440 = 24 hours
                # #"-l", '-mppmem={0}'.format(self.memory),               # Max memory per cpu in MB (strict - attempts to allocate more memory will fail)
                # "-e", "logs/outerror." + block_name('_')[0],      # Error out file
                # "-j", "eo"]                                            # Join standard out file to error file

            # Better to use file input rather than command line inputs (according to XSEDE helpdesk)
            # Request MAX_CORES so that memory requirement is also met
            full_command += (
               "#PBS -N {0}\n".format(block_name) +
               "#PBS -A hvd113\n" +
               "#PBS -q {0}\n".format(QSUB_WORK_QUEUE) +
               "#PBS -l nodes=1:ppn={0}:native,walltime={1}:00\n".format(str(MAX_CORES), self.required_full_time) +
               "#PBS -e logs/outerror.{0}\n".format(block_name.split('_')[0]) +
               "#PBS -j eo\n")

        if len(dependency_set) > 0:
            if USE_SBATCH:
                dependency_string = ":".join(d for d in dependency_set)
                if len(dependency_string) > 0:
                    print "depends on jobs:" + dependency_string
                    command_list += ["-d", "afterok:" + dependency_string]
            elif USE_QSUB:
                dependency_string = ":".join(d for d in dependency_set)
                if len(dependency_string) > 0:
                    print "depends on jobs:" + dependency_string
                    full_command += "#PBS -W depend=afterok:" + dependency_string + "\n"
            else:
                command_list += " && ".join("done(%s)" % d for d in dependency_set)

        if USE_SBATCH:
            full_command += "date\n"
        elif USE_QSUB:
            full_command += "cd $PBS_O_WORKDIR\ndate\n"

        # Generate job block commands
        block_commands = ''
        for j in self.job_block_list:
            block_commands += '{0} &\n'.format(' '.join(j.command()))
            print j.name
        full_command += '{0}wait\ndate\n'.format(block_commands)

        # # Test job ids
        # for job_block in job_block_list:
        #     for j in job_block:
        #         j.jobid = str(cls.block_count - 1)

        # print "command_list: {0}".format(command_list)
        # print "full_command: {0}".format(full_command)

        # Submit job
        process = subprocess.Popen(command_list, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        submit_out, submit_err = process.communicate(full_command)

        # print submit_out
        # print submit_err

        # Process output
        if len(submit_err) == 0:
            if USE_SBATCH:
                new_jobid = submit_out.split()[3]
            elif USE_QSUB:
                new_jobid = submit_out.split('.')[0]
            print 'jobid={0}'.format(new_jobid)
            for j in self.job_block_list:
                j.jobid = new_jobid


    @classmethod
    def run_all_job_blocks(cls):
        print "Running all job blocks ({0} blocks)".format(len(cls.all_job_blocks))
        for job_block in cls.all_job_blocks:
            job_block.submit_block()



class JobBlockOrganizer(object):
    # In-charge of orgainizing all the jobs into JobBlocks while ensuring that all dependencies are being met
    def __init__(self):
        # A per threads number job blocks list
        self.job_blocks_per_thread_lists = {}

    def add_job(self, job):
        if not job.get_threads_num() in self.job_blocks_per_thread_lists.keys():
            self.job_blocks_per_thread_lists[job.get_threads_num()] = []

        # Iterate over the job blocks lists of this threads number, and find a job block where this job can be added to
        inserted = False
        for job_block in self.job_blocks_per_thread_lists[job.get_threads_num()]:
            if job_block.can_add_job(job):
                job_block.add_job(job)
                return

        # if no suitable block was found, create a new block, and add the job to it
        new_job_block = JobBlock(job.get_threads_num())
        self.job_blocks_per_thread_lists[job.get_threads_num()].append(new_job_block)
        new_job_block.add_job(job)

    def run_all(self):
        JobBlock.run_all_job_blocks()

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
        self.time = 20
        self.is_java_job = True
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
        self.memory = 4000
        self.time = 10
        self.is_java_job = True
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
        self.memory = 4000
        self.time = 10
        self.is_java_job = True
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
    def __init__(self, dependencies, tiles_fname1, tiles_fname2, ransac_fname, image_width, image_height, fixed_layers, pmcc_output_file, jar_file, conf_fname=None, threads_num=1, auto_add_model=False):
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
        if auto_add_model:
            self.auto_add_model = '--auto_add_model'
        else:
            self.auto_add_model = ''
        self.threads = threads_num
        self.threads_str = '-t {0}'.format(threads_num)
        self.dependencies = dependencies
        self.memory = 7000
        self.time = 30
        self.is_java_job = True
        self.output = pmcc_output_file
        #self.already_done = os.path.exists(self.output_file)

    def command(self):
        return ['python',
                os.path.join(os.environ['ALIGNER'], 'scripts', 'match_layers_by_max_pmcc.py'),
                self.output_file, self.fixed_layers, self.jar_file, self.conf_fname, self.threads_str, self.auto_add_model, self.image_width, self.image_height,
                self.tiles_fname1, self.tiles_fname2, self.ransac_fname]


                ##job_pmcc = MatchLayersByMaxPMCC(dependencies, layers_data[i]['ts'], layers_data[i + j]['ts'], \
                ##    layers_data[i]['ransac'][i + j], imageWidth, imageHeight, \
                ##    [ fixed_layer ], pmcc_fname, args.jar_file, conf_fname=args.conf_file_name)
                ###match_layers_by_max_pmcc(args.jar_file, layer_to_ts_json[i], layer_to_ts_json[i + j], ransac_fname, imageWidth, imageHeight, [fixed_layer], pmcc_fname, conf)


class OptimizeLayersElastic(Job):
    def __init__(self, dependencies, outputs, tiles_fnames, corr_fnames, image_width, image_height, fixed_layers, output_dir, jar_file, conf_fname=None):
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
        self.memory = 4000
        self.time = 30
        self.is_java_job = True
        self.output = outputs
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
    parser.add_argument('--auto_add_model', action="store_true", 
                        help='automatically add the identity model, if a model is not found')
    parser.add_argument('-k', '--keeprunning', action='store_true', 
                        help='Run all jobs and report cluster jobs execution stats')
    parser.add_argument('-m', '--multicore', action='store_true', 
                        help='Run all jobs in blocks on multiple cores')

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

    all_running_jobs = []


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
        layers_to_process = min(i + args.max_layer_distance + 1, all_layers[-1] + 1) - i
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
                all_running_jobs.append(job_match)
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
                all_running_jobs.append(job_ransac)
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
                    [ fixed_layer ], pmcc_fname, args.jar_file, conf_fname=args.conf_file_name, threads_num=8, auto_add_model=args.auto_add_model)
                #match_layers_by_max_pmcc(args.jar_file, layer_to_ts_json[i], layer_to_ts_json[i + j], ransac_fname, imageWidth, imageHeight, [fixed_layer], pmcc_fname, conf)
                pmcc_jobs.append(job_pmcc)
                all_running_jobs.append(job_pmcc)
            layers_data[si]['matched_pmcc'][sij] = pmcc_fname

            all_pmcc_files.append(pmcc_fname)



    print "all_pmcc_files: {0}".format(all_pmcc_files)

    # Optimize all layers to a single 3d image
    create_dir(args.output_dir)
    sections_outputs = []
    for i in all_layers:
        out_section = os.path.join(args.output_dir, "Section{0}.json".format(str(i).zfill(3)))
        sections_outputs.append(out_section)

    dependencies = all_running_jobs
    job_optimize = OptimizeLayersElastic(dependencies, sections_outputs, all_ts_files, all_pmcc_files, \
        imageWidth, imageHeight, [ fixed_layer ], args.output_dir, args.jar_file, conf_fname=args.conf_file_name)
    #optimize_layers_elastic(all_ts_files, all_pmcc_files, imageWidth, imageHeight, [fixed_layer], args.output_dir, args.jar_file, conf)


    # Run all jobs
    if args.keeprunning:
        Job.keep_running()
    elif args.multicore:
        # Bundle jobs for multicore nodes
        # if RUN_LOCAL:
        #     print "ERROR: --local cannot be used with --multicore (not yet implemented)."
        #     sys.exit(1)
        Job.multicore_run_all()
    else:
        Job.run_all()

