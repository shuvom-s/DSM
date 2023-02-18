import sys
import gzip
import numpy as np

hapsfile = sys.argv[1]
nvar = sys.argv[2]
nsamp = sys.argv[3]
outprefix = sys.argv[4]

# out_samples = outprefix + "_samples_haplo.txt"
# out_var = outprefix + "_var_haplo.txt"
out_mat = outprefix + "_mat.txt"




# fsamp = open(out_samples,'w')
# fvar = open(out_var,'w')
# fmat = open(out_mat,'w')

genomap = {'0|0':'0', '0|1':'1', '1|1':'2', '1|0':'1'}
haplomap1 = {'0|0': 0, '0|1': 0, '1|1': 1, '1|0': 1, "0/0": 0, "1/1": 1, "0/1": -1}
haplomap2 = {'0|0': 0, '0|1': 1, '1|1': 1, '1|0': 0, "0/0": 0, "1/1": 1, "0/1": -1}


def parsesnp_dense(s):
    geno = s.split(":")[0]
    if geno not in genomap:
    	return '-1'
    else:
    	return genomap[geno]

def parsesnp_dense_haplo(s):
    geno = s.split(":")[0]
    if geno not in haplomap1:
        return (-1,-1)
    else:
        return (haplomap1[geno], haplomap2[geno])



vindex = 1
# for line in open(vcffile):
print('nvar: ', int(nvar))
print('nsamp: ', int(nsamp))
mat_geno = np.zeros((int(nvar), int(nsamp)*2), dtype=np.int8)
print(mat_geno.shape)

with open(hapsfile, 'rt') as f:
    # lines = [line.rstrip() for line in f]
    for i, line in enumerate(f):
        line = line.rstrip()
        tok = line.split(" ")

        to_write = np.array(list(map(int,tok[5:])))
        tw = np.where((to_write==0)|(to_write==1), to_write^1, to_write)
        mat_geno[i,:] = tw

        if i % 10000 == 0:
            print("{} lines processed".format(i), end="\r")




np.save(out_mat, mat_geno)

