import numpy as np
import pandas as pd
import argparse
from copy import deepcopy
import sys
from scipy.stats.stats import pearsonr  
import math




        
        
def ebl(genos, prob_mat, gene_exp, connectivity, corr_thresh=0.45, extremity_thresh=0.95, train_test_split=588):
    '''
    Hybrid EBL model which collapses to GNB for non-extreme SNPs. For original, pass in matrix of zeros as prob mat and set extremity posteriors to
    1.
    '''
    prob_mat_copy = deepcopy(prob_mat)
    gene_exp_FUSION = gene_exp[train_test_split:,:]

    genos_FUSION = genos[:,train_test_split:]
    
    # print(connectivity)
    
    gene_exp_GTEX = gene_exp[:train_test_split,:]
    genos_GTEX = genos[:,:train_test_split]
    

    gexp_train = gene_exp_GTEX[:,connectivity[1,:]]
    genos_train = genos_GTEX[connectivity[0,:],:].T
    
    corrs = np.zeros(gexp_train.shape[1])
    for i in range(len(corrs)):
        corr = pearsonr(genos_train[:,i].flatten(), gexp_train[:,i].flatten())[0]
        # print(corr)
        corrs[i] = corr
        # if i % 1000 == 0: print(corr)
    
    df = pd.DataFrame({"Geno": connectivity[0,:], "Exp": connectivity[1,:], "Corr": corrs})
    df['AbsCorr'] = abs(df['Corr'])

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
    
    
    y_train_np = genos[:,:train_test_split]
    # print(y_train_np.shape)
    
    freqs = [((y_train_np == 0).sum(axis=1)+1)/(train_test_split+3),
         ((y_train_np == 1).sum(axis=1)+1)/(train_test_split+3),
         ((y_train_np == 2).sum(axis=1)+1)/(train_test_split+3)]
    
   
    prob_mat = np.array(freqs).T
    # print(prob_mat.shape)
    prob_mat = np.log(np.repeat(prob_mat[np.newaxis, :, :], genos_FUSION.shape[1], axis=0))
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
        # print(prob_mat.shape)
        to_replace =  prob_mat[list(above_thresh_idx),geno_idx, :] 
        # print(to_replace.shape)
        to_replace[:,:,0] = np.log(np.ones(to_replace.shape[0]) * 0.001)
        to_replace[:,:,1] = np.log(np.ones(to_replace.shape[0]) * 0.009)
        to_replace[:,:,2] = np.log(np.ones(to_replace.shape[0]) * 0.99)
        # print(prob_mat.shape)
        # print(above_thresh_idx)
        
        prob_mat[list(above_thresh_idx), geno_idx,:] = to_replace
        
        to_replace =  prob_mat[list(below_thresh_idx),geno_idx,:] 
        to_replace[:,:,0] = np.log(np.ones(to_replace.shape[0]) * 0.99)
        to_replace[:,:,1] = np.log(np.ones(to_replace.shape[0]) * 0.009)
        to_replace[:,:,2] = np.log(np.ones(to_replace.shape[0]) * 0.001)
        
        prob_mat[list(below_thresh_idx), geno_idx,:] = to_replace
    # print(prob_mat)
    return prob_mat

            
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chrom', help='chromosome number')
    parser.add_argument('--corr', help='correlation threshold', type=float)
    parser.add_argument('--extremity', help='extremity threshold', type=float)
    parser.add_argument('--eqtl_path', help='path to eqtl csv')
    parser.add_argument('--prob_mat', help='probability matrix')
    parser.add_argument('--gene_exp', help='gene expression path')
    parser.add_argument('--connectivity', help='connectivity matrix path')
    parser.add_argument('--genos', help='genotype path')
    parser.add_argument('--ebl_path', help='path to save ebl probability matrix')
    parser.add_argument('--train_test_split', help='split index between training and testing')
    # path_to_FUSION = "/broad/cholab/dfridman/FUSION_study_data/phg001194.v1.FUSION_TisssueBiopsy2_Imputation.genotype-calls-vcf.c1/processed_geno_files"
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

    genos = np.load(args.genos)
    prob_mat = np.load(args.prob_mat)
    
    
    gene_exp = np.load(args.gene_exp)
    connectivity = np.load(args.connectivity)
    
    ebl_mat = ebl(genos, prob_mat, gene_exp, connectivity, corr, extremity)

    np.save(args.ebl_path, ebl_mat)

    
    
    
if __name__ == "__main__":
    main()