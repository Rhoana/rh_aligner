import sys
import cPickle
import pylab
from matplotlib.collections import LineCollection


if __name__ == "__main__":
    displacements = cPickle.load(open(sys.argv[1]))
    x, y = zip(*displacements.keys())
    for k1, k2 in displacements:
        if 1440 <= k1 <= 1450:
            p1, p2 = displacements[k1, k2]
            pylab.figure()
            pylab.plot(p1[:, 0], p1[:, 1], '+r')
            lines = [[a, a + 10 * (b - a)] for a, b in zip(p1, p2)]
            ls = LineCollection(lines, linestyles='solid')
            pylab.gca().add_collection(ls)
            pylab.title('{} to {}'.format(k1, k2))
    pylab.show()
