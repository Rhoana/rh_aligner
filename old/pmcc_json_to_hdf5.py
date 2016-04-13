import h5py
import ujson
import numpy as np
import sys
import os
import multiprocessing as mp
import glob
import utils

def convert(file_in, file_out):
    data = ujson.load(open(file_in, "r"))
    assert not os.path.exists(file_out)
    with h5py.File(file_out, 'w') as hf:
        for idx, m in enumerate(data):
            hf.create_dataset("matches{}_url1".format(idx),
                              data=np.array(m["url1"].encode("utf-8"), dtype='S'))
            hf.create_dataset("matches{}_url2".format(idx),
                              data=np.array(m["url2"].encode("utf-8"), dtype='S'))

            p1s = np.array([(pair["p1"]["l"][0], pair["p1"]["l"][1])
                            for pair in m["correspondencePointPairs"]],
                           dtype=np.float32)
            p2s = np.array([(pair["p2"]["l"][0], pair["p2"]["l"][1])
                            for pair in m["correspondencePointPairs"]],
                           dtype=np.float32)
            hf.create_dataset("matches_{}_p1".format(idx), data=p1s)
            hf.create_dataset("matches_{}_p2".format(idx), data=p2s)

def create_chunks(l, n):
    sub_size = max(1, (len(l) - 1)/n + 1)
    return [l[i:i + sub_size] for i in range(0, len(l), sub_size)]

def convert_files(files_list, json_dir, hdf5_dir):
    for f in files_list:
        in_file = os.path.join(json_dir, f)
        out_file = os.path.join(hdf5_dir, "{}.hdf5".format(os.path.splitext(os.path.basename(f))[0]))
        if not os.path.exists(out_file):
            convert(in_file, out_file)

def parallel_convert(processes_num, json_dir, hdf5_dir):
    utils.create_dir(hdf5_dir)

    # Get the json files list
    json_files = glob.glob(os.path.join(json_dir, '*pmcc.json'))

    # create N-1 worker processes
    if processes_num > 1:
        pool = mp.Pool(processes=processes_num - 1)
    print "Creating {} other processes, and parsing {} json files".format(processes_num - 1, len(json_files))

    # Divide the list into processes_num chunks
    chunks = create_chunks(json_files, processes_num)

    async_res = None

    # run all jobs but one by other processes
    for sub_list in chunks[:-1]:
        async_res = pool.apply_async(convert_files, (sub_list, json_dir, hdf5_dir))

    # run the last job by the current process
    print "running last list with {} files".format(len(chunks[-1]))
    convert_files(chunks[-1], json_dir, hdf5_dir)

    # wait for all other processes to finish their job
    if processes_num > 1:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == '__main__':
    processes_num = int(sys.argv[1])
    json_dir = sys.argv[2]
    hdf5_dir = sys.argv[3]
    parallel_convert(processes_num, json_dir, hdf5_dir)

