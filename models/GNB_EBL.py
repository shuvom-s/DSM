#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:23:47 2020

@author: dfridman
"""
import pandas as pd
import numpy as np
import scipy
import torch
from scipy.special import logsumexp
from scipy import stats
from copy import deepcopy
import argparse
from scipy.stats.stats import pearsonr 


def corr_coeff(A,B) :
    # Get number of rows in either A or B
    N = B.shape[0]

    # Store columnw-wise in A and B, as they would be used at few places
    sA = A.sum(0)
    sB = B.sum(0)

    # Basically there are four parts in the formula. We would compute them one-by-one
    p1 = N*np.einsum('ij,ik->kj',A,B)
    p2 = sA*sB[:,None]
    p3 = N*((B**2).sum(0)) - (sB**2)
    p4 = N*((A**2).sum(0)) - (sA**2)

    # Finally compute Pearson Correlation Coefficient as 2D array
    pcorr = ((p1 - p2)/np.sqrt(p4*p3[:,None]))

    # Get the element corresponding to absolute argmax along the columns
#   out = pcorr[np.nanargmax(np.abs(pcorr),axis=0),np.arange(pcorr.shape[1])]

    return pcorr

# Function to count #samples/class
def class_counts(genotype_class, pseudo=1):
    class_0_count = np.sum(genotype_class.detach().numpy() == 0, axis=0)+pseudo
    class_1_count = np.sum(genotype_class.detach().numpy() == 1, axis=0)+pseudo
    class_2_count = np.sum(genotype_class.detach().numpy() == 2, axis=0)+pseudo
    return dict({0: class_0_count, 1: class_1_count, 2: class_2_count})

def edge_case_count_reset(geno_count):
    # Temporarily set missing genotype count to 1 to avoid error in mean calculation
    # print(np.where(geno_count[2] == 1))
    edge_case_idx_geno0 = np.where(geno_count[0] <= 3)[0]
    edge_case_idx_geno1 = np.where(geno_count[1] <= 3)[0]
    edge_case_idx_geno2 = np.where(geno_count[2] <= 3)[0]
    
    edge_case_count_geno0 = []
    edge_case_count_geno1 = []
    edge_case_count_geno2 = []
    
    edge_case_idx = [edge_case_idx_geno0, edge_case_idx_geno1, edge_case_idx_geno2]
    edge_case_count = [edge_case_count_geno0, edge_case_count_geno1, edge_case_count_geno2]
    
    for i in range(3):
        edge_case_count[i] = geno_count[i][edge_case_idx[i]]
        geno_count[i][edge_case_idx[i]] = 3
    return edge_case_idx, edge_case_count, geno_count

def edge_case_renormalization(genotype_log_prob_normalized, edge_case_idx, edge_case_count):
    genotype_prob_normalized = np.exp(genotype_log_prob_normalized)
    for i in range(3):
        genotype_prob_normalized[:, edge_case_idx[i], i] = (edge_case_count[i] + 1)/(genotype_prob_normalized.shape[0] + 3)
        if i == 0:
            j,k = 1,2
        elif i == 1:
            j,k = 0,2
        elif i == 2:
            j,k = 0,1
        genotype_prob_normalized[:, edge_case_idx[i], j] = genotype_prob_normalized[:, edge_case_idx[i], j]/(genotype_prob_normalized[:, edge_case_idx[i], j] + genotype_prob_normalized[:, edge_case_idx[i], k] + (edge_case_count[i] + 1)/(genotype_prob_normalized.shape[0] + 3))
        genotype_prob_normalized[:, edge_case_idx[i], k] = genotype_prob_normalized[:, edge_case_idx[i], k]/(genotype_prob_normalized[:, edge_case_idx[i], j] + genotype_prob_normalized[:, edge_case_idx[i], k] + (edge_case_count[i] + 1)/(genotype_prob_normalized.shape[0] + 3))
    # for i in range(3):
    #     print((genotype_prob_normalized[:,edge_case_idx[i],0] + genotype_prob_normalized[:,edge_case_idx[i],1] + genotype_prob_normalized[:,edge_case_idx[i],2]))
    genotype_log_prob_normalized = np.log(genotype_prob_normalized)
    return genotype_log_prob_normalized
   
    

# log-Normal distribution for matrix with different mean and variance by column
def log_Norm_dist(X_mat, mu_mat, sigma_mat):
    norm_factor = -np.log(sigma_mat) -0.5*np.log(2*np.pi)
    Gauss_func = ((X_mat[None,:,:] - mu_mat[None,:,:].transpose((2,0,1)))**2) / (2*(sigma_mat[None,:,:].transpose((1,0,2)))**2)
    # Gauss_func = ((X_mat.reshape([1, X_mat.shape[0], X_mat.shape[1]]) - 
    #             mu_mat.reshape([mu_mat.shape[1], 1, mu_mat.shape[0]]))**2)/(
    #             2*(sigma_mat.reshape([sigma_mat.shape[0], 1, sigma_mat.shape[1]]))**2)
    log_prob = norm_factor[None,:,:].transpose((1,0,2)) - Gauss_func
    # log_prob = norm_factor.reshape([norm_factor.shape[0], 1, norm_factor.shape[1]]) - Gauss_func.numpy()
    return log_prob



# Function outputting Bayesian log(p(genotype|expression))
def Bayesian_model_fit_2(y_train, X_train, y_test, X_test, connectivity_mat):
    '''
    Make sure to allocate enough memory for this function. This is the GNB model, which uses a Gaussian Naive Bayes assumption.
    '''
    eGene_idx_mat = np.zeros((X_test.shape[1], y_test.shape[1]))
    eGene_idx_mat[connectivity_mat[1], connectivity_mat[0]] = 1 
    eGene_idx_mat = torch.tensor(eGene_idx_mat.T).unsqueeze(1)

    
    geno_count = class_counts(y_train,0)
    pseudo = 1
    tot_count = geno_count[0] + geno_count[1] + geno_count[2] + 3 * pseudo 
    geno_0_prob = (geno_count[0] + pseudo) / tot_count
    geno_1_prob = (geno_count[1] + pseudo) / tot_count
    geno_2_prob = (geno_count[2] + pseudo) / tot_count 
    
    # geno_0_prob = geno_0_freq/(geno_0_freq + geno_1_freq + geno_2_freq)
    # geno_1_prob = geno_1_freq/(geno_0_freq + geno_1_freq + geno_2_freq)
    # geno_2_prob = geno_2_freq/(geno_0_freq + geno_1_freq + geno_2_freq)
    
    geno_0_sample_mat = (y_train.detach().numpy() == 0).astype(int)
    geno_1_sample_mat = (y_train.detach().numpy() == 1).astype(int)
    geno_2_sample_mat = (y_train.detach().numpy() == 2).astype(int)
    
    # geno_0_mean_exp = np.dot(X_train.T, geno_0_sample_mat)/X_train.shape[0]
    # geno_1_mean_exp = np.dot(X_train.T, geno_1_sample_mat)/X_train.shape[0]
    # geno_2_mean_exp = np.dot(X_train.T, geno_2_sample_mat)/X_train.shape[0]

    # Temporarily set missing genotype count to 1 to avoid error in mean calculation
    # print(np.where(geno_count[2] == 1))
    # edge_case_idx_geno0 = np.where(geno_count[0] <= 3)[0]
    # edge_case_idx_geno1 = np.where(geno_count[1] <= 3)[0]
    # edge_case_idx_geno2 = np.where(geno_count[2] <= 3)[0]
    
    # edge_case_idx = [edge_case_idx_geno0, edge_case_idx_geno1, edge_case_idx_geno2]
    
    # edge_case_count_geno0 = geno_count[0][edge_case_idx_geno0]
    # edge_case_count_geno1 = geno_count[1][edge_case_idx_geno1]
    # edge_case_count_geno2 = geno_count[2][edge_case_idx_geno2]
    
    # Set count arbitrarily to 3 (not 0 to avoid divide by zero error and not 1 to avoid exp - mean = 0 for case with only one sample)
    # geno_count[2][edge_case_idx] = 3
    edge_case_idx, edge_case_count, geno_count = edge_case_count_reset(geno_count)
    
    geno_0_mean_exp = np.dot(X_train.T, geno_0_sample_mat)/geno_count[0]
    geno_1_mean_exp = np.dot(X_train.T, geno_1_sample_mat)/geno_count[1]
    geno_2_mean_exp = np.dot(X_train.T, geno_2_sample_mat)/geno_count[2]
    
    geno_0_diff = X_train[None,:,:] - geno_0_mean_exp[None,:,:].transpose((2,0,1))
    geno_1_diff = X_train[None,:,:] - geno_1_mean_exp[None,:,:].transpose((2,0,1))
    geno_2_diff = X_train[None,:,:] - geno_2_mean_exp[None,:,:].transpose((2,0,1))
    
    # geno_0_diff = X_train.reshape([1, X_train.shape[0], X_train.shape[1]]) - geno_0_mean_exp.reshape(
    #                                 [geno_0_mean_exp.shape[1], 1, geno_0_mean_exp.shape[0]])
    # geno_1_diff = X_train.reshape([1, X_train.shape[0], X_train.shape[1]]) - geno_1_mean_exp.reshape(
    #                                 [geno_1_mean_exp.shape[1], 1, geno_1_mean_exp.shape[0]])
    # geno_2_diff = X_train.reshape([1, X_train.shape[0], X_train.shape[1]]) - geno_2_mean_exp.reshape(
    #                                 [geno_2_mean_exp.shape[1], 1, geno_2_mean_exp.shape[0]])
    
    geno_0_sample_idx_mat = np.repeat(geno_0_sample_mat.transpose()[:,:,np.newaxis], geno_0_diff.shape[2], axis=2)
    geno_0_sample_idx_mat_sum = np.sum(geno_0_sample_idx_mat, axis=1)
    geno_0_sample_idx_mat_sum[edge_case_idx[0], :] = 1
    geno_0_exp_sum = np.sum(geno_0_diff.numpy()**2 * geno_0_sample_idx_mat, axis=1)
    geno_0_exp_sum[edge_case_idx[0], :] = 1
    geno_0_std_exp = np.sqrt(geno_0_exp_sum/geno_0_sample_idx_mat_sum)
    
    geno_1_sample_idx_mat = np.repeat(geno_1_sample_mat.transpose()[:,:,np.newaxis], geno_1_diff.shape[2], axis=2)
    geno_1_sample_idx_mat_sum = np.sum(geno_1_sample_idx_mat, axis=1)
    geno_1_sample_idx_mat_sum[edge_case_idx[1], :] = 1
    geno_1_exp_sum = np.sum(geno_1_diff.numpy()**2 * geno_1_sample_idx_mat, axis=1)
    geno_1_exp_sum[edge_case_idx[1], :] = 1
    geno_1_std_exp = np.sqrt(geno_1_exp_sum/geno_1_sample_idx_mat_sum)
    
    geno_2_sample_idx_mat = np.repeat(geno_2_sample_mat.transpose()[:,:,np.newaxis], geno_2_diff.shape[2], axis=2)
    geno_2_sample_idx_mat_sum = np.sum(geno_2_sample_idx_mat, axis=1)
    geno_2_sample_idx_mat_sum[edge_case_idx[2], :] = 1
    geno_2_exp_sum = np.sum(geno_2_diff.numpy()**2 * geno_2_sample_idx_mat, axis=1)
    geno_2_exp_sum[edge_case_idx[2], :] = 1
    geno_2_std_exp = np.sqrt(geno_2_exp_sum/geno_2_sample_idx_mat_sum)
    
    # geno_0_sample_idx_mat = np.repeat(geno_0_sample_mat.transpose()[:,:,np.newaxis], geno_0_diff.shape[2], axis=2)
    # geno_0_std_exp = np.sqrt(np.sum(geno_0_diff.numpy()**2 * geno_0_sample_idx_mat, axis=1)/geno_count[0])
    # geno_1_sample_idx_mat = np.repeat(geno_1_sample_mat.transpose()[:,:,np.newaxis], geno_1_diff.shape[2], axis=2)
    # geno_1_std_exp = np.sqrt(np.sum(geno_1_diff.numpy()**2 * geno_1_sample_idx_mat, axis=1)/geno_count[1])
    # geno_2_sample_idx_mat = np.repeat(geno_2_sample_mat.transpose()[:,:,np.newaxis], geno_2_diff.shape[2], axis=2)
    # geno_2_std_exp = np.sqrt(np.sum(geno_2_diff.numpy()**2 * geno_2_sample_idx_mat, axis=1)/geno_count[2])
    
    # geno_0_std_exp = np.sqrt(np.sum(geno_0_diff.numpy()**2, axis=1)/y_train.shape[0])
    # geno_1_std_exp = np.sqrt(np.sum(geno_1_diff.numpy()**2, axis=1)/y_train.shape[0])
    # geno_2_std_exp = np.sqrt(np.sum(geno_2_diff.numpy()**2, axis=1)/y_train.shape[0])
    
    # Extract only relevant probabilities (from eGenes)
    geno_0_log_Norm = torch.tensor(log_Norm_dist(X_test.numpy(), geno_0_mean_exp, geno_0_std_exp)) * eGene_idx_mat
    geno_1_log_Norm = torch.tensor(log_Norm_dist(X_test.numpy(), geno_1_mean_exp, geno_1_std_exp)) * eGene_idx_mat
    geno_2_log_Norm = torch.tensor(log_Norm_dist(X_test.numpy(), geno_2_mean_exp, geno_2_std_exp)) * eGene_idx_mat
    
    Prob_geno_0 = geno_0_log_Norm.sum(axis=2).T + np.log(geno_0_prob)
    Prob_geno_1 = geno_1_log_Norm.sum(axis=2).T + np.log(geno_1_prob)
    Prob_geno_2 = geno_2_log_Norm.sum(axis=2).T + np.log(geno_2_prob)
    
    genotype_log_prob = torch.stack((Prob_geno_0, Prob_geno_1, 
                                        Prob_geno_2), axis=-1)
    z = np.expand_dims(logsumexp(genotype_log_prob, axis=-1), axis=-1)
    genotype_log_prob_normalized = genotype_log_prob.numpy() - z
    
    genotype_log_prob_normalized = edge_case_renormalization(genotype_log_prob_normalized, edge_case_idx, edge_case_count)
    
    # genotype_prob_normalized = np.exp(genotype_log_prob_normalized)
    # genotype_prob_normalized[:, edge_case_idx[2], 2] = (edge_case_count[2] + 1)/(genotype_prob_normalized.shape[0] + 3)
    # genotype_prob_normalized[:, edge_case_idx[2], 0] = genotype_prob_normalized[:, edge_case_idx[2], 0]/(genotype_prob_normalized[:, edge_case_idx[2], 0] + genotype_prob_normalized[:, edge_case_idx[2], 1] + (edge_case_count[2] + 1)/(genotype_prob_normalized.shape[0] + 3))
    # genotype_prob_normalized[:, edge_case_idx[2], 1] = genotype_prob_normalized[:, edge_case_idx[2], 1]/(genotype_prob_normalized[:, edge_case_idx[2], 0] + genotype_prob_normalized[:, edge_case_idx[2], 1] + (edge_case_count[2] + 1)/(genotype_prob_normalized.shape[0] + 3))

    
    # genotype_prob_normalized = np.exp(genotype_log_prob_normalized)
    # genotype_prob_normalized[:, edge_case_idx, 2] = (edge_case_count + 1)/(genotype_prob_normalized.shape[0] + 3)
    # genotype_prob_normalized[:, edge_case_idx, 0] = genotype_prob_normalized[:, edge_case_idx, 0]/(genotype_prob_normalized[:, edge_case_idx, 0] + genotype_prob_normalized[:, edge_case_idx, 1] + (edge_case_count + 1)/(genotype_prob_normalized.shape[0] + 3))
    # genotype_prob_normalized[:, edge_case_idx, 1] = genotype_prob_normalized[:, edge_case_idx, 1]/(genotype_prob_normalized[:, edge_case_idx, 0] + genotype_prob_normalized[:, edge_case_idx, 1] + (edge_case_count + 1)/(genotype_prob_normalized.shape[0] + 3))
    # print((genotype_prob_normalized[:,edge_case_idx[2],0] + genotype_prob_normalized[:,edge_case_idx[2],1] + genotype_prob_normalized[:,edge_case_idx[2],2]))

    # genotype_log_prob_normalized = np.log(genotype_prob_normalized)
    
    return genotype_log_prob_normalized


def run_chrom_geno(gene_exp, connectivity, genos, window_start, train_test_split, window_end=None):
    genos = genos.T
    # gene_exp = gene_exp.T
    x_train = torch.Tensor(gene_exp[0:train_test_split,:])
    x_test = torch.Tensor(gene_exp[train_test_split:,:])
    
    
    # prob_mat = np.load("Bayesian_mats/chr{}/chr{}_prob_mat_GTEX.npy".format(chrom, chrom))
    if window_end is not None:
        y_train = torch.Tensor(genos[0:train_test_split,window_start:window_end])
        y_test = torch.Tensor(genos[train_test_split:,window_start:window_end])
        # prob_mat = prob_mat[:,window_start:window_end,:]
        connectivity_ub = connectivity[:,connectivity[0,:] < window_end]
        connectivity_lb = connectivity_ub[:,connectivity_ub[0,:] >= window_start]
        
    else:
        y_train = torch.Tensor(genos[0:train_test_split,window_start:])
        y_test = torch.Tensor(genos[train_test_split:train_test_split,window_start:])
        # prob_mat = prob_mat[:,window_start:window_end,:]
        connectivity_lb = connectivity[:,connectivity[0,:] >= window_start]
    
    
    
    # nsnps = size
    connectivity_lb[0,:] = connectivity_lb[0,:] - window_start

    mat = Bayesian_model_fit_2(y_train, x_train, y_test, x_test, connectivity_lb)
    return mat

       
        
def ebl(genos, prob_mat, gene_exp, connectivity, corr_thresh=0.45, extremity_thresh=0.95, train_test_split=588):
    '''
    This is a hybrid EBL model which collapses to GNB for SNPs which do not meet the extremity thresholds. For original, only predict extreme SNPs.
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
    # parser.add_argument('--chrom', help='chromosome number')
    parser.add_argument('--genos', help='genotype path')
    parser.add_argument('--connectivity', help='connectivity matrix path')
    parser.add_argument('--gene_exp', help='gene expression path')
    parser.add_argument('--save_path', help='save path for gnb/ebl')
    parser.add_argument('--train_test_split', help='split index between training and testing')
    
    parser.add_argument('--corr', help='correlation threshold', type=float, default=0.45)
    parser.add_argument('--extremity', help='extremity threshold', type=float, default=0.9)
    parser.add_argument('--eqtl_path', help='path to eqtl csv')
    parser.add_argument('--prob_mat', help='probability matrix')
    parser.add_argument('--method', help='gnb or ebl', default="gnb")
    
    args = parser.parse_args()
    
    genos = np.load(args.genos)
    
    connectivity = np.load(args.connectivity)
    train_test_split = args.train_test_split

    # geneexp = np.load(args.gene_exp)
    
    exp = np.load(args.gene_exp)
    # connectivity = np.load(connectivity_mat)

    if args.method is "gnb":
        window_size = 2500
        nwindows = genos.shape[0] // window_size

        mats = []
        for window in range(nwindows+1):
            window_start = window*window_size

            if window != nwindows:
                window_end = (window+1)*window_size
                mat = run_chrom_geno(exp, connectivity, genos, window_start, int(train_test_split), window_end=window_end)
            else:
                mat = run_chrom_geno(exp, connectivity, genos, window_start, int(train_test_split))
            print(mat.shape)
            mats.append(mat)
        total_mat = np.concatenate(mats, axis=1)

        np.save(args.save_path, total_mat)
    
    
    else:
        if args.corr:
            corr = args.corr
        if args.extremity:
            extremity = args.extremity
        genos = np.load(args.genos)
        prob_mat = np.load(args.prob_mat)


        gene_exp = np.load(args.gene_exp)
        connectivity = np.load(args.connectivity)

        ebl_mat = ebl(genos, prob_mat, gene_exp, connectivity, corr, extremity)

        np.save(args.save_path, ebl_mat)

    
if __name__ == "__main__":
    main()
    
    

# def run_chrom(chrom, gene_exp, connectivity, genos, window_start, window_end=None):
#     # print(gene_exp.shape)
#     # print(connectivity.shape)
#     print(genos.shape)
#     genos = genos.T
#     # gene_exp = gene_exp.T
#     x_train = torch.Tensor(gene_exp[0:450,:])
#     x_test = torch.Tensor(gene_exp[450:588,:])
    
#     expand_xtrain = torch.zeros(2*x_train.shape[0], x_train.shape[1])
#     expand_xtrain[::2, :] = x_train   # Index every second row, starting from 0
#     expand_xtrain[1::2, :] = x_train
    
#     expand_xtest = torch.zeros(2*x_test.shape[0], x_test.shape[1])
#     expand_xtest[::2, :] = x_test   # Index every second row, starting from 0
#     expand_xtest[1::2, :] = x_test
    
    
#     if window_end is not None:
#         y_train = torch.Tensor(genos[0:900,window_start:window_end])
#         y_test = torch.Tensor(genos[900:1176,window_start:window_end])
#         connectivity_ub = connectivity[:,connectivity[0,:] < window_end]
#         connectivity_lb = connectivity_ub[:,connectivity_ub[0,:] >= window_start]
        
#     else:
#         y_train = torch.Tensor(genos[0:900,window_start:])
#         y_test = torch.Tensor(genos[900:1176,window_start:])
#         connectivity_lb = connectivity[:,connectivity[0,:] >= window_start]
    
    
    
#     # nsnps = size
#     connectivity_lb[0,:] = connectivity_lb[0,:] - window_start
    
    
#     # print(connectivity_lb)
#     mat = baseline.Bayesian_model_fit_haplo(y_train, expand_xtrain, y_test, expand_xtest, connectivity_lb)
#     return mat
    
# 