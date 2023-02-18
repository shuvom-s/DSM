#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 18:12:29 2021

@author: dfridman
"""

import pandas as pd
import numpy as np
import torch

def file_loader(gene_exp_path, geno_mat_path, geno_var_info_path, eQTL_path, add_cov : bool, cov_path = None):
    print('loading files (Daniel)')
    Gene_Exp = pd.read_csv(gene_exp_path, sep='\t', compression='gzip')
    Geno_Mat = np.load(geno_mat_path)
    Geno_var_info = pd.read_csv(geno_var_info_path, sep='\t', compression='gzip')
    eQTL = pd.read_csv(eQTL_path, sep='\t', compression='gzip')
    if add_cov:
        Cov = pd.read_csv(cov_path, sep='\t', compression='gzip', index_col='ID')
    else:
        Cov = None
    return Gene_Exp, Geno_Mat, Geno_var_info, eQTL, Cov

# Function for filtering data by p-value and MAF
def Genotype_filtering(Geno_mat_mtx, Geno_var_df, eQTL_df, eQTL_version, Gene_Exp_df, gene_corr, maf_thresh, p_thresh: bool, pval=None):
    print("Filtering genotypes...")
    if p_thresh:
        print("Using p thresholding")
        filtered_eQTL = eQTL_df[eQTL_df['pval_nominal'] < pval]
        print(filtered_eQTL)
    else:
        # Use p_val_thresh = None for significant eQTL data, specify threshold for all eQTL data
        filtered_eQTL = eQTL_df
    
    # Extract SNPs based on MAF (!!! ADD BACK IN !!!)
    # filtered_eQTL = filtered_eQTL[(filtered_eQTL['maf'] >= maf_thresh[0]) & (filtered_eQTL['maf'] < maf_thresh[1])]
    
    #Filter genotype matrix such that all SNPs contain no more than 10% missing samples
    SNP_include = ((Geno_mat_mtx == -1).mean(axis=1) < 0.1)
    Geno_mat_mtx = Geno_mat_mtx[SNP_include, :]
    Geno_var_df = Geno_var_df.loc[SNP_include,:]
    
    # Filter eQTL to include only genes present in Gene_Exp dataframe
    filtered_eQTL = filtered_eQTL[filtered_eQTL['gene_id'].isin(Gene_Exp_df['gene_id'])]
    # Filter only eSNPs below specified p-value threshold
    if eQTL_version == 'v8':
        var_bool = Geno_var_df['ID'].isin(filtered_eQTL['variant_id'])
    elif eQTL_version == 'v7' or 'v6p' or 'v6':
        # filtered_eQTL = filtered_eQTL[filtered_eQTL['gene_id'].isin(Gene_Exp_df['gene_id'])]
        var_bool = Geno_var_df['variant_id_hg19'].isin(filtered_eQTL['variant_id'])
    else:
        raise 'Must provide valid GTEx version (v6, v6p, v7, or v8)'
    var_idx = np.array(Geno_var_df.loc[var_bool].index)
    idx = np.where(var_bool == True)[0]
    filtered_Geno_var = Geno_var_df.iloc[idx, :]
    filtered_Geno_mat = Geno_mat_mtx[idx,:]
    if eQTL_version == 'v8':
        filtered_eQTL = filtered_eQTL[filtered_eQTL['variant_id'].isin(filtered_Geno_var['ID'])]
        maf = filtered_Geno_var.merge(filtered_eQTL.drop_duplicates(subset=['variant_id'])[['variant_id', 'maf']], how='left', left_on='ID', right_on='variant_id')
    elif eQTL_version == 'v7' or 'v6p' or 'v6':
        filtered_eQTL = filtered_eQTL[filtered_eQTL['variant_id'].isin(filtered_Geno_var['variant_id_hg19'])]
        maf = filtered_Geno_var.merge(filtered_eQTL.drop_duplicates(subset=['variant_id'])[['variant_id', 'maf']], how='left', left_on='variant_id_hg19', right_on='variant_id')
    if gene_corr:
        filtered_Gene_Exp = Gene_Exp_df
    else:
        filtered_Gene_Exp = Gene_Exp_df[Gene_Exp_df['gene_id'].isin(np.unique(filtered_eQTL['gene_id']))]
    return filtered_Geno_mat, filtered_Geno_var, filtered_eQTL, filtered_Gene_Exp, var_idx, maf['maf']
    # return filtered_Geno_mat, filtered_Geno_var, filtered_eQTL, filtered_Gene_Exp, var_idx

# Function for selecting random variants
def rand_target(Geno_mat_mtx, Geno_var_df, eQTL_df, eQTL_version, Gene_Exp_df, n_variants: int):
    # target_idx = np.random.randint(0, Geno_mat_df.shape[0], n_variants)
    target_idx = np.random.choice(Geno_mat_mtx.shape[0], size=n_variants, replace=False)
    rand_geno_var_info = Geno_var_df.iloc[target_idx]
    # rand_geno_mat = Geno_mat_df.iloc[target_idx, :]
    rand_geno_mat = Geno_mat_mtx[target_idx, :]
    if eQTL_version == 'v8':
        rand_eQTL = eQTL_df[eQTL_df['variant_id'].isin(rand_geno_var_info['ID'])]
    elif eQTL_version == 'v7' or 'v6p' or 'v6':
        rand_eQTL = eQTL_df[eQTL_df['variant_id'].isin(rand_geno_var_info['variant_id_hg19'])]
    # rand_Gene_Exp = Gene_Exp_df[Gene_Exp_df['gene_id'].isin(np.unique(rand_eQTL['gene_id']))]
    rand_Gene_Exp = Gene_Exp_df
    return rand_geno_mat, rand_geno_var_info, rand_eQTL, rand_Gene_Exp

# Create data tensors    
def data_tensors(Gene_Exp, Geno_mat, add_cov: bool, cov=None):
    input_tensor_Exp = torch.tensor(np.array(Gene_Exp.iloc[:,1:])).T
    # target_tensor = torch.tensor(np.array(Geno_mat.astype(int)), dtype=torch.int64).T
    # target_tensor = torch.tensor(np.array(Geno_mat).astype(float)).T
    target_tensor = torch.tensor(np.array(Geno_mat).astype(float)).transpose(0,1)
    if add_cov:
        input_tensor_cov = torch.tensor(np.array(cov.astype(float))).T.double()
    else:
        input_tensor_cov = None
    return input_tensor_Exp, target_tensor, input_tensor_cov

# Create sparse connectivity matrix 
def connectivity_matrix(Gene_Exp_df, eQTL_df, eQTL_version, Geno_var_df, gene_corr=None):
    all_eGene_idx = []
    for var_idx in range(Geno_var_df.shape[0]):
        if var_idx % 5000 == 0:
            print('percent processed: ', round(var_idx/Geno_var_df.shape[0], 2))
        if eQTL_version == 'v8':
            var_ID = Geno_var_df['ID'].iloc[var_idx]
        elif eQTL_version == 'v7' or 'v6p' or 'v6':
            var_ID = Geno_var_df['variant_id_hg19'].iloc[var_idx]
        if gene_corr is None:
            eSNP_eGene_idx = np.where(Gene_Exp_df['gene_id'].isin(eQTL_df[eQTL_df['variant_id']==var_ID]['gene_id']))[0]
        else:
            eSNP_eGene_idx = np.where(Gene_Exp_df['gene_id'].isin(eQTL_df[eQTL_df['variant_id']==var_ID]['gene_id']))[0]
            # print(eSNP_eGene_idx)
            eSNP_eGene_idx = np.where(np.abs(gene_corr[eSNP_eGene_idx,:]) > 0.25)[1]
            # print(eSNP_eGene_idx)
        all_eGene_idx.append(eSNP_eGene_idx)
    # Create connectivity matrix (non-zero rows and columns) for sparsity matrix
    row_vec = []
    for idx, array in enumerate(all_eGene_idx):
        if idx % 1000 == 0:
            print('percent processed: ', round(idx/len(all_eGene_idx), 2)) 
        row_idx = np.repeat(idx, len(array))
        row_vec.append(row_idx)
        
    sparse_col_vec = np.concatenate(all_eGene_idx)
    sparse_row_vec = np.concatenate(row_vec)
    # count #eGenes per SNP
    _,eGene_count = np.unique(sparse_row_vec, return_counts=True)
    print(len(eGene_count))
    print('avg gene count: ', np.round(np.mean(eGene_count), 3))
    # connectivity matrix with 1st row -> row index of connection, 2nd row -> column index of connection
    connectivity_mat = np.stack([sparse_row_vec, sparse_col_vec], axis=0)
    connectivity_mat = torch.LongTensor(connectivity_mat)
    return connectivity_mat, eGene_count


