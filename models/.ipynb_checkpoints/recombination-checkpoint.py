import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import argparse



def recombinations(pval, chrom, dataset, map_path, eqtl_path):
    df = pd.read_csv(map_path, compression='gzip', sep='\t')
    
    eqtl_info = pd.read_csv(eqtl_path, compression='gzip', sep='\t')
    
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
    # parser.add_argument('--chrom', help='chromosome')
    parser.add_argument('--pval', help='pval threshold')
    parser.add_argument('--dataset', help='dataset')
    parser.add_argument('--outpath', help='output path')
    parser.add_argument('--map_path', help='path to genetic map')
    parser.add_argument('--eqtl_path', help='path to eqtl info')
    args = parser.parse_args()
    
    
    dataset = str(args.dataset)
    if not args.pval:
        pval = 'all'
    
    if not args.chrom:
        for chrom in range(19,23):
            eqtl_df = recombinations(pval, chrom, dataset)
            eqtl_df.to_csv(args.outpath)
    else:
        eqtl_df = recombinations(pval, chrom, dataset, args.map_path, args.eqtl_path)
    
if __name__ == '__main__':
    main()
        
    
    
