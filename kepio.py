import gzip
import random
import zipfile

def unzip_data():
    with zipfile.ZipFile('small.zip', 'r') as small:
        small.extractall('build')

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
        if adj.has_key(i):
            adj[i].append(j)
        else:
            adj[i] = [j]
        w[i,j] = float(data.pop(0))
        assert i>=0 and j>=0
    return adj, w

def write_prob(filename):
    def rnd():
        return random.random()
    
    adj, w = read_kep(filename)
    filename = filename.replace(".input", ".prob")
    if filename.find(".gz") > 0:
        f = gzip.open(filename,"w")
    else:
        f = open(filename,"w")

    for i in adj:
        f.write("%s\n" % rnd())
        for j in adj[i]:
            f.write("%s " % rnd())  
        f.write("\n")

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
            p[i,j] = float(data.pop(0))

    return adj, w, p

# if __name__ == "__main__":
#     import sys
#     import os 
#     try:
#         filename = sys.argv[1]
#         seed = int(sys.argv[2])
#     except:
#         filename = "DATA/10-instance-01.input.gz"
#         seed = 1

#     random.seed(seed)
#     probfile = filename.replace(".input", ".prob")
#     if os.path.exists(probfile) or os.path.exists(probfile+".gz"):
#         adj, w, p = read_prob(filename)
#     else:
#         write_prob(filename)
#         adj, w, p = read_prob(filename)

    # for i in adj:
    #     print i, "\t", adj[i]
    #  
    # for i in adj:
    #     print p[i]
    #     for j in adj[i]:
    #         print p[i,j],
    #     print

