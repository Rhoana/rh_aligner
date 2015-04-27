import h5py
import ujson
import numpy as np
import sys
import os.path

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

if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
