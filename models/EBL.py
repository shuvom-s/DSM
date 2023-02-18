import numpy as np
import pandas as pd
import argparse
from copy import deepcopy
import sys
from scipy.stats.stats import pearsonr  
import math


def harmanci_uniform():
    pass


        
        
def harmanci(genos, prob_mat, gene_exp, connectivity, corr_thresh=0.45, extremity_thresh=0.95):
    prob_mat_copy = deepcopy(prob_mat)
    gene_exp_FUSION = gene_exp[588:,:]
    # print(gene_exp_FUSION.shape)
    genos_FUSION = genos[:,588:]
    
    print(connectivity)
    
    gene_exp_GTEX = gene_exp[:588,:]
    genos_GTEX = genos[:,:588]
    

    gexp_train = gene_exp_GTEX[:,connectivity[1,:]]
    genos_train = genos_GTEX[connectivity[0,:],:].T
    
    corrs = np.zeros(gexp_train.shape[1])
    for i in range(len(corrs)):
        corr = pearsonr(genos_train[:,i].flatten(), gexp_train[:,i].flatten())[0]
        # print(corr)
        corrs[i] = corr
        if i % 1000 == 0: print(corr)
    # corrs = np.corrcoef(gexp_train.T, genos_train.T)
    df = pd.DataFrame({"Geno": connectivity[0,:], "Exp": connectivity[1,:], "Corr": corrs})
    df['AbsCorr'] = abs(df['Corr'])
    # print(df[df['Corr'] > 0.35])
    idx = df.groupby(['Geno'])['AbsCorr'].transform(max) == df['AbsCorr']
    df_filtered = df[idx]
    df_thresh = df_filtered[(df_filtered['Corr'] > corr_thresh) | (df_filtered['Corr'] < -corr_thresh)]
    
    subset = df_thresh[['Geno', 'Exp']]
    print(df_thresh)
    tuples = [tuple(x) for x in subset.to_numpy()]
    # print(tuples)
    
    quantileuppers = []
    quantilelowers = []
    for idx in tuples:
        gene = gene_exp_GTEX[:,idx[1]]
        order = gene.argsort()
        ranks = gene.argsort()
        quantileuppers.append(np.quantile(gene, extremity_thresh))
        quantilelowers.append(np.quantile(gene, 1-extremity_thresh))
        
        
        # extremity = ranks/588 - 0.5
        # print(extremity)
    df_thresh['UpperQuantile'] = quantileuppers
    df_thresh['LowerQuantile'] = quantilelowers
    # print(df_thresh[df_thresh['Corr'] > 0])
    
    
    y_train_np = genos[:,:588]
    # print(y_train_np.shape)
    
    freqs = [((y_train_np == 0).sum(axis=1)+1)/(588+3),
         ((y_train_np == 1).sum(axis=1)+1)/(588+3),
         ((y_train_np == 2).sum(axis=1)+1)/(588+3)]
    
   
    prob_mat = np.array(freqs).T
    # print(prob_mat.shape)
    prob_mat = np.log(np.repeat(prob_mat[np.newaxis, :, :], 292, axis=0))
    # prob_mat_copy = deepcopy(prob_mat)
    
    corrs = list(df_thresh['Corr'])
    qus = list(df_thresh['UpperQuantile'])
    qls = list(df_thresh['LowerQuantile'])
    genos = list(df_thresh['Geno'])
    exps = list(df_thresh['Exp'])
    
    for i, corr in enumerate(corrs):
        exp = gene_exp_FUSION[:,exps[i]]
        
        if corr >= 0:
            above_thresh_idx = np.where(exp > qus[i])
            below_thresh_idx = np.where(exp < qls[i])
        else:
            above_thresh_idx = np.where(exp < qls[i])
            below_thresh_idx = np.where(exp > qus[i])
        
        geno_idx = genos[i]
        print(prob_mat.shape)
        to_replace =  prob_mat[list(above_thresh_idx),geno_idx, :] 
        print(to_replace.shape)
        to_replace[:,:,0] = np.log(np.ones(to_replace.shape[0]) * 0.001)
        to_replace[:,:,1] = np.log(np.ones(to_replace.shape[0]) * 0.009)
        to_replace[:,:,2] = np.log(np.ones(to_replace.shape[0]) * 0.99)
        print(prob_mat.shape)
        print(above_thresh_idx)
        
        prob_mat[list(above_thresh_idx), geno_idx,:] = to_replace
        
        to_replace =  prob_mat[list(below_thresh_idx),geno_idx,:] 
        to_replace[:,:,0] = np.log(np.ones(to_replace.shape[0]) * 0.99)
        to_replace[:,:,1] = np.log(np.ones(to_replace.shape[0]) * 0.009)
        to_replace[:,:,2] = np.log(np.ones(to_replace.shape[0]) * 0.001)
        
        prob_mat[list(below_thresh_idx), geno_idx,:] = to_replace
    return prob_mat
                # to_replace[0,:,:] = np.array
            # print(prob_mat[extreme_idx, list(connectivity_new.flatten())[1], :])
            
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chrom', help='chromosome number')
    parser.add_argument('--corr', help='correlation threshold', type=float)
    parser.add_argument('--extremity', help='extremity threshold', type=float)
    path_to_FUSION = "/broad/cholab/dfridman/FUSION_study_data/phg001194.v1.FUSION_TisssueBiopsy2_Imputation.genotype-calls-vcf.c1/processed_geno_files"
    args = parser.parse_args()
    
    chrom = 20
    if args.chrom:
        chrom = args.chrom
    extremity = 0.95
    if args.extremity:
        extremity = args.extremity
    corr = 0.45
    if args.corr:
        corr = args.corr
    path = "/broad/cholab/shuvom/HMMs/inputs_and_matrices/chr{}/".format(chrom)
    
    # genotypes = path_to_FUSION + "/chr{}_geno_mat.npy".format(chrom)
    
    # eqtl_genos = np.load(path + "chr{}_GTEx_FUSION_merged_haplos_eqtls.npy".format(eqtl_genos))
    # # print(eqtl_genos.shape)
    eqtl_all = np.load(path + "chr{}_GTEx_FUSION_merged_haplos_eqtls.npy".format(chrom))
    genos = eqtl_all.reshape(eqtl_all.shape[0], 880, 2).sum(axis=2)
    prob_mat = np.load("Bayesian_mats/chr{}/chr{}_prob_mat_gtexonly.npy".format(chrom, chrom))
    
    # path = "/broad/cholab/shuvom/HMMs/inputs_and_matrices/chr{}/".format(chrom)
    eqtls = path_to_FUSION + "/eqtls/all_eqtls_chr{}_FUSION_haplo_mat.npy".format(chrom)
    gene_exp = np.load(path + "gene_exp_chr{}.npy".format(chrom))
    connectivity = np.load(path + "connectivity_chr{}.npy".format(chrom))
    
    eqtl_info = pd.read_csv("/broad/cholab/dfridman/ordinal_model_code/func_geno_privacy/data/GTEx_v8_Muscle_Skeletal_eQTL_by_chr/GTEx_EUR_eQTL_chr{}.txt.gz".format(chrom), sep='\t', compression='gzip')
    all_snp_info = pd.read_csv("/broad/cholab/dfridman/ordinal_model_code/func_geno_privacy/data/processed_geno_files/GTEx_v8_FUSION_chr{}_merged_geno_var.txt.gz".format(chrom), sep='\t', compression='gzip')
    eqtl_w_recombs = pd.read_csv("/broad/cholab/shuvom/HMMs/inputs_and_matrices/chr{}/all_eqtls_chr{}_FUSION_recombinations.txt.gz".format(chrom, chrom), sep='\t', compression='gzip')
    
    eqtl_snp_info = all_snp_info[all_snp_info['Pos'].isin(eqtl_w_recombs['Pos'])].drop_duplicates(subset='Pos', keep='first').reset_index(drop=True)
    
    path = "/broad/cholab/shuvom/HMMs/inputs_and_matrices/chr{}/".format(chrom)
    eqtl_haplos = np.load(path + "chr{}_GTEx_FUSION_merged_haplos_eqtls.npy".format(chrom))
    
    prob_mat_copy = deepcopy(prob_mat)
    harmanci_mat = harmanci(genos, prob_mat, gene_exp, connectivity, corr, extremity)
    print(np.sum(np.abs(np.exp(prob_mat_copy) - np.exp(harmanci_mat))))
    np.save("Bayesian_mats/chr{}/chr{}_prob_mat_harmanci_gtexonly.npy".format(chrom, chrom,
                                                                                         corr, extremity), 
            harmanci_mat)

    # np.save("Bayesian_mats/chr{}/chr{}_prob_mat_harmanciflipped.npy".format(chrom, chrom), 
    #         harmanci_mat)
    
    
    
if __name__ == "__main__":
    main()