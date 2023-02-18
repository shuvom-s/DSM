import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import argparse

path_to_data = '/broad/cholab/dfridman/ordinal_model_code/func_geno_privacy/data/processed_geno_files'
path_to_fusion_snps = "/broad/cholab/dfridman/FUSION_study_data/phg001194.v1.FUSION_TisssueBiopsy2_Imputation.genotype-calls-vcf.c1/processed_geno_files/"
path_to_matrices = '/broad/cholab/shuvom/HMMs/inputs_and_matrices/'


def recombinations(pval, chrom, dataset):
    df = pd.read_csv('/broad/cholab/shuvom/GLIMPSE/maps/genetic_maps.b37/chr{}.b37.gmap.gz'.format(chrom), compression='gzip', sep='\t')
    
    if dataset == 'gtex':
        eqtl_info = pd.read_csv(path_to_fusion_snps +\
                                'eqtls/{}_eqtls_chr{}_FUSION_filtered_samples.txt.gz'.format(pval, 
                                                                                             chrom),
                                compression='gzip', sep='\t')
        
    if dataset == 'geuvadis':
        eqtl_info = pd.read_csv("/broad/cholab/shuvom/GTExFUSION/Geuvadis/geuvadis_chr20_PHASED_var_haplo_UNINDEX_FULL.txt", sep='\t')
     
    
    f = interp1d(df['pos'], df['cM'], fill_value="extrapolate")
    eqtl_info['cM'] = f(eqtl_info['Pos'])
    
    haplos = [500,1000,2000,5000,10000]
    
    for nhaplos in haplos:
        a_param = nhaplos/(4 * 2000)
        next_pos = np.array(list(eqtl_info['cM'])[1:])
        prev_pos = np.array(list(eqtl_info['cM'])[:-1])
        pos_diff = next_pos - prev_pos
        recombs = 1-np.exp(-pos_diff/a_param)
        recombs[recombs < 1e-5] = 1e-5
        recombs = [0] + list(recombs)

        eqtl_info['recombination_neffective{}'.format(nhaplos)] = recombs
    
    # print(eqtl_info)
    return eqtl_info
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chrom', help='chromosome')
    parser.add_argument('--pval', help='pval threshold')
    parser.add_argument('--dataset', help='dataset')
    parser.add_argument('--outpath', help='output path')
    args = parser.parse_args()
    
    
    dataset = str(args.dataset)
    if not args.pval:
        pval = 'all'
    
    if not args.chrom:
        for chrom in range(19,23):
            eqtl_df = recombinations(pval, chrom, dataset)
            eqtl_df.to_csv(args.outpath)
if __name__ == '__main__':
    main()
        
    
    
