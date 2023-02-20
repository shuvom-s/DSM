import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--chrom', help='chromosome number', type=int)
parser.add_argument('--subset', help='path to file of indices of haplotype file to subset', type=int)
parser.add_argument('--outpath', help='path to output', type=int)
parser.add_argument('--inpath', help='path to output', type=int)
args = parser.parse_args()

chrom=19
if args.chrom:
    chrom = args.chrom

subset = pd.read_csv(args.subset, header=None)
subset.rename(columns={0:"pos"}, inplace=True)


lines_to_keep = list(subset["pos"])


out = open(args.outpath, "w")

with open(args.inpath) as f:
    for i, line in enumerate(f):
        if i % 10000 == 0:
            print("{} lines processed".format(i))
        if i in lines_to_keep:
            out.write(line)

f.close()
out.close()