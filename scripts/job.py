# A Generic job class, that is extended by a specific job class that needs to be executed on the cluster

import sys
import os.path
import os
import subprocess
import datetime
import time
from itertools import product
from collections import defaultdict

LOGS_DIR = './logs'
MEASURE_PERFORMANCE = False
RUN_LOCAL = False
USE_QSUB = False
USE_SBATCH = True

# Make sure the logs directory exists
if not os.path.exists(LOGS_DIR) or not os.path.isdir(os.path.dirname(LOGS_DIR)):
    os.makedirs(LOGS_DIR)



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
                "--no-requeue",
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

    @classmethod
    def multicore_run_list(cls, jobs):
        all_jobs_optimizer = JobBlockOrganizer()

        for j in jobs:

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

        submitted_job_blocks = all_jobs_optimizer.run_all()
        return submitted_job_blocks

    @classmethod
    def multicore_run_all(cls):
        multicore_run_list(cls.all_jobs)

    @classmethod
    def multicore_keep_running(cls):
        all_jobs_complete = False
        cancelled_jobs = {}
        cancelled_requeue_iters = 5
        submitted_job_blocks = {}

        while not all_jobs_complete:

            # Find running job blocks
            sacct_output = subprocess.check_output(['sacct', '-n', '-o', 'JobID,JobName%100,State%20'])

            pending_running_complete_job_blocks = {}
            pending = 0
            running = 0
            complete = 0
            failed = 0
            cancelled = 0
            timeout = 0
            other_status = 0
            non_matching = 0

            timed_out_jobs = set()

            for job_line in sacct_output.split('\n'):

                job_split = job_line.split()
                if len(job_split) == 0:
                    continue

                job_id = job_split[0]
                job_name = job_split[1]
                job_status = ' '.join(job_split[2:])
                
                if job_name in submitted_job_blocks:
                    if job_status in ['PENDING', 'RUNNING', 'COMPLETED']:
                        if job_name in pending_running_complete_job_blocks:
                            print 'Found duplicate job: ' + job_name
                            dup_job_id, dup_job_status = pending_running_complete_job_blocks[job_name]
                            print job_id, job_status, dup_job_id, dup_job_status

                            job_to_kill = None
                            if job_status == 'PENDING':
                                job_to_kill = job_id
                            elif dup_job_status == 'PENDING':
                                job_to_kill = dup_job_id
                                pending_running_complete_job_blocks[job_name] = (job_id, job_status)    

                            if job_to_kill is not None:
                                print 'Canceling job ' + job_to_kill
                                try:
                                    scancel_output = subprocess.check_output(['scancel', '{0}'.format(job_to_kill)])
                                    print scancel_output
                                except:
                                    print "Error canceling job:", sys.exc_info()[0]
                        else:
                            pending_running_complete_job_blocks[job_name] = (job_id, job_status)
                            if job_status == 'PENDING':
                                pending += 1
                            elif job_status == 'RUNNING':
                                running += 1
                            elif job_status == 'COMPLETED':
                                complete += 1
                    elif job_status in ['FAILED', 'NODE_FAIL']:
                        failed += 1
                    elif job_status in ['CANCELLED', 'CANCELLED+'] or job_status.startswith('CANCELLED'):
                        cancelled += 1

                        # This job could requeued after preemption
                        # Wait cancelled_requeue_iters before requeueing
                        cancelled_iters = 0
                        if job_id in cancelled_jobs:
                            cancelled_iters = cancelled_jobs[job_id]

                        if cancelled_iters < cancelled_requeue_iters:
                            pending_running_complete_job_blocks[job_name] = (job_id, job_status)
                            cancelled_jobs[job_id] = cancelled_iters + 1

                    elif job_status in ['TIMEOUT']:
                        timeout += 1
                        # in case of a timeout, add all jobs to the timed_out_jobs set
                        job_block_list = submitted_job_blocks[job_name]
                        timed_out_jobs.update(job_block_list)
                    else:
                        print "Unexpected status: {0}".format(job_status)
                        other_status += 1
                elif job_name not in ['batch', 'true', 'prolog']:
                    non_matching += 1

            #print 'Found {0} running job blocks.'.format(len(pending_running_complete_job_blocks))

            # Find running jobs
            pending_running_complete_jobs = {}
            for job_block_name in pending_running_complete_job_blocks:
                job_id, job_status = pending_running_complete_job_blocks[job_block_name]
                job_block_list = submitted_job_blocks[job_block_name]
                for job in job_block_list:
                    pending_running_complete_jobs[job.name] = (job_id, job_status)

            #print '== {0} running jobs.'.format(len(pending_running_complete_jobs))

            # Make a list of runnable jobs
            run_count = 0
            block_count = 0
            runnable_jobs = []
            for j in cls.all_jobs:
                if j.name not in pending_running_complete_jobs and not j.get_done() and j.dependendencies_done():
                    # if the job is now available to run, and was previously timed out, then increase its time
                    if j in timed_out_jobs:
                        j.time *= 2
                        print "Extending the time of job: {0} to {1} minutes".format(j.name, j.time)
                    runnable_jobs.append(j)
                    run_count += 1

            new_job_blocks = Job.multicore_run_list(runnable_jobs)
            block_count += len(new_job_blocks)
            submitted_job_blocks.update(new_job_blocks)

            print 'Found {0} pending, {1} running, {2} complete, {3} failed, {4} cancelled, {5} timeout, {6} unknown status and {7} non-matching job blocks.'.format(
                pending, running, complete, failed, cancelled, timeout, other_status, non_matching)

            print "Queued {0} job{1} in {2} block{3}.".format(
                run_count, '' if run_count == 1 else 's',
                block_count, '' if block_count == 1 else 's')

            if pending > 0 or running > 0 or run_count > 0:
                time.sleep(60)
            else:
                all_jobs_complete = True


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
        submitted_job_blocks = {}
        # If there are no jobs, do nothing
        if self.jobs_count == 0:
            return submitted_job_blocks

        # If th ejob block was already submitted, do nothing
        if not self.pending:
            return submitted_job_blocks

        self.pending = False

        # receursively execute all the dependencies
        for other_job_block in self.job_block_dependencies:
            if other_job_block.is_pending():
                submitted_job_blocks.update(other_job_block.submit_block())

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
                "--no-requeue",
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

        submitted_job_blocks[block_name] = self.job_block_list
        return submitted_job_blocks

    # @classmethod
    # def run_all_job_blocks(cls):
    #     print "Running all job blocks ({0} blocks)".format(len(cls.all_job_blocks))
    #     for job_block in cls.all_job_blocks:
    #         job_block.submit_block()



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
        submitted_job_blocks = {}
        for threads in self.job_blocks_per_thread_lists.keys():
            for job_block in self.job_blocks_per_thread_lists[threads]:
                submitted_job_blocks.update(job_block.submit_block())
        #JobBlock.run_all_job_blocks()
        return submitted_job_blocks
