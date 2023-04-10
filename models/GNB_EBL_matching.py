



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chrom', help='chromosome number')
    parser.add_argument('--prob_mat', help='probability matrix')
    parser.add_argument('--gene_exp', help='gene expression path')
    parser.add_argument('--connectivity', help='connectivity matrix path')
    parser.add_argument('--haplos', help='haplotype path')
    parser.add_argument('--genosnum', help='number of genotypes')
    parser.add_argument('--save_path', help='path to save match matrix')
    
    args = parser.parse_args()
    #chromosome
    chrom = args.chrom
    genosnum = args.genosnum
    prob_mat_path = args.prob_mat
    haplos_path = args.haplos
    match_path = args.save_path


    # matrix of p(x|e)
    # change the argument of np.load() to wherever your p(x|e) matrix is
    # chr_prob_mat = torch.tensor(np.load("/broad/cholab/shuvom/Bayesian_baseline/Bayesian_mats/chr{}/chr{}_prob_mat_GTEX.npy".format(chrom, chrom)))
    chr_prob_mat = torch.tensor(np.load(prob_mat_path))

    chr_prob_mat = torch.nan_to_num(chr_prob_mat, nan=0.0, posinf=0, neginf=0)

    # FUSION genotypes
    chr_haplos = np.load()
    chr_genos = chr_haplos.reshape(chr_haplos.shape[0], genosnum, 2).sum(axis=2)

    one_hot = torch.transpose(torch.nn.functional.one_hot(torch.tensor(chr_genos).to(torch.int64)),
                                 1,0)

    scores = (chr_prob_mat) * one_hot

    correct_scores = scores.sum(axis=2).sum(axis=1)

    # initialize match matrix
    matches = np.empty((genosnum, chr_prob_mat.shape[0]))
    matches[0,:] = correct_scores.cpu().detach().numpy()


    for i in range(1,292):
        # roll the probability matrix p(x|e) forward
        rolled = torch.roll(one_hot, i, 0)

        scores = (chr_prob_mat) * rolled

        incorrect_scores = scores.sum(axis=2).sum(axis=1)
        matches[i,:] = incorrect_scores.cpu().detach().numpy()
    
    np.save(matches, match_path)