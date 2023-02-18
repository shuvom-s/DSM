import numpy as np
import pandas as pd
import argparse

# allgenos_info = pd.read_csv("/broad/cholab/shuvom/ref_panels/raw/chr20.info.24hrs.txt", sep='\t', header=None)

# allgenos_info.rename(columns={1: "pos"}, inplace=True)

# eqtls_info = pd.read_csv("/broad/cholab/shuvom/HMMs/inputs_and_matrices/chr20/Geno_var_info_filt.tsv", sep='\t')
# subset=allgenos_info[allgenos_info['pos'].isin(eqtls_info['Pos'])].drop_duplicates(subset=['pos'], keep='first')
parser = argparse.ArgumentParser()
parser.add_argument('--chrom', help='chromosome number', type=int)
args = parser.parse_args()
chrom=19
if args.chrom:
    chrom = args.chrom

subset = pd.read_csv("/broad/cholab/shuvom/ref_panels/chr{}/eqtlidx.txt".format(chrom), header=None)
subset.rename(columns={0:"pos"}, inplace=True)
print(subset)

lines_to_keep = list(subset["pos"])
# print(lines_to_keep)

out = open("/broad/cholab/shuvom/ref_panels/chr{}/negatives_eqtls.haps".format(chrom), "w")

with open("/broad/cholab/shuvom/ref_panels/chr{}/chr{}.fakes.haps".format(chrom, chrom)) as f:
    for i, line in enumerate(f):
        if i % 10000 == 0:
            print("{} lines processed".format(i))
        if i in lines_to_keep:
            out.write(line)

f.close()
out.close()