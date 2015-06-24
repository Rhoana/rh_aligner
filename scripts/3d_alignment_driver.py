import sys
import json
import subprocess
from multiprocessing.pool import Pool

def prelimmatchingworker(cmd):
    p = subprocess.Popen(cmd);
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
            if abs(slice1 - slice2) <= numforwardback:
                commands.append("python slice_to_slice_comparison.py " + str(slice1) + " " + str(slice2) + " " + datadir + " " + workdir + " " + conffile)

    pool = Pool(processes=10);
    results = [pool.apply_async(prelimmatchingworker, [cmd]) for cmd in commands];

if __name__ == '__main__':
    main()
