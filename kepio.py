import gzip
import zipfile


def unzip_data():
    with zipfile.ZipFile("small.zip", "r") as small:
        small.extractall("build")


def read_kep(filename):
    if filename.find(".gz") > 0:
        f = gzip.open(filename)
    else:
        f = open(filename)
    data = f.read().split()
    nvert = int(data.pop(0))
    narcs = int(data.pop(0))
    adj = {}
    w = {}
    for a in range(narcs):
        i = int(data.pop(0))
        j = int(data.pop(0))
        if i in adj:
            adj[i].append(j)
        else:
            adj[i] = [j]
        w[i, j] = float(data.pop(0))
        assert i >= 0 and j >= 0
    return adj, w


def read_prob(filename):
    adj, w = read_kep(filename)
    filename = filename.replace(".input", ".prob")
    if filename.find(".gz") > 0:
        f = gzip.open(filename)
    else:
        f = open(filename)

    data = f.read().split()
    p = {}
    for i in adj:
        p[i] = float(data.pop(0))
        for j in adj[i]:
            p[i, j] = float(data.pop(0))

    return adj, w, p
