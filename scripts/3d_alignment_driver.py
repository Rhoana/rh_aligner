import sys
import json
import subprocess
import multiprocessing
import os
from subprocess import PIPE

def prelimmatchingworker(cmd):
    p = subprocess.Popen([cmd], stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    p.wait()

def main():
    script, conffile = sys.argv
    with open(conffile) as conf_file:
        conf = json.load(conf_file)
    slicemin = conf["driver_args"]["slicemin"]
    slicemax = conf["driver_args"]["slicemax"]
    numforwardback = conf["driver_args"]["numforwardback"]
    datadir = conf["driver_args"]["datadir"]
    workdir = conf["driver_args"]["workdir"]
    conffile = conf["driver_args"]["conffile"]
    
    commands = []
    # First pass for preliminary mfov matching
    for slice1 in range(slicemin, slicemax + 1):
        for slice2 in range(slicemin, slicemax + 1):
            if abs(slice1 - slice2) <= numforwardback and slice1 != slice2:
                commands.append("python /home/raahilsha/slice_to_slice_comparison.py " + str(slice1) + " " + str(slice2) + " " + datadir + " " + workdir + " " + conffile)

    pool = multiprocessing.Pool(10)
    pool.map(prelimmatchingworker, commands);

if __name__ == '__main__':
    main()
