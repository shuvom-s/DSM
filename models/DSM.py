import torch
import numpy as np
from copy import deepcopy
import pandas as pd
from torch_sparse import spmm
import argparse
import sys
import os
import time


def loss_fn(p_xe, p_xe_negative=None, contrastive=False):
    if not contrastive:
        loss = -torch.sum(p_xe)
    else:
        loss = -torch.sum(p_xe) + torch.sum(p_xe_negative)
    return loss


class OrdinalHMM(torch.nn.Module):
    def __init__(self, predictor: torch.nn.Module, ref_tensor: torch.Tensor, 
                recombination_rates: torch.Tensor) -> None:
        super().__init__()
        self.predictor = deepcopy(predictor)
        self.ref_tensor = ref_tensor
        self.recomb_prob = torch.log(recombination_rates)
        self.eprob = 0.01
        pseudocount = 1.0 
        self.emission_dosage_haplo = np.array([[(1-self.eprob), self.eprob],
                                            [self.eprob, (1-self.eprob)]])
        self.emission_dosage_haplo += pseudocount / (self.ref_tensor.shape[1])

    def forward_pass_hmm(self, x_onehots):
        # ref tensor should be m x n
        # m = nsnps, n = ref panel size
        m = int(self.ref_tensor.shape[0])
        n = int(self.ref_tensor.shape[1])
        # print("one hot shape")
        # print(x_onehots.shape)
        
        assert int(x_onehots.shape[1]) == m
        batch_size = int(x_onehots.shape[0])
        
        log_emission_dosage_haplo = torch.log(torch.tensor(self.emission_dosage_haplo)).unsqueeze(0).repeat(batch_size, 1, 1)

        # set up comutation matrices
        fprobx_haplo = torch.zeros(batch_size, m, n)
        p_hmm = torch.zeros(batch_size)

        # forward
        for i in range(m):
            # if i % 10 == 0:
            #     print(i, "SNPS completed, forward")
            # get reference panel for this snp
            ref_dosage_haplo = (self.ref_tensor[i][None,:]).repeat(batch_size, 1)

            # set up matrix to store emission matrix
            state_emission_haplo = torch.zeros(batch_size, n, 2)

            # for both dosages, fill emission matrix
            for dos in range(2):
                emissions = (ref_dosage_haplo == dos)[:,:,None] * log_emission_dosage_haplo[:,dos,:][:,None,:]
                state_emission_haplo += emissions

            # dosage prob shape should be batch size x 2
            dosage_prob_arg = fprobx_haplo[:,i,:][:,:,None] + state_emission_haplo
            dosage_prob = torch.logsumexp(dosage_prob_arg, dim=(1,))
            dosage_prob -= torch.logsumexp(dosage_prob, dim=(1,))[:,None]
            # dosage_prob = logsumexp(dosage_prob_arg, dim=(1,))
            # dosage_prob -= logsumexp(dosage_prob_arg, dim=(1,))[:,None]
            

            # get agreement between forward probx and one hot encoded haplotype, with state emission in between
            p_hmm_update = torch.sum(dosage_prob * x_onehots[:,i,:], dim = 1)
            p_hmm += p_hmm_update


            if i == m-1:
                break

            # update next prob
            nextprob = fprobx_haplo[:,i,:] + torch.einsum('ikl, il -> ik', state_emission_haplo, x_onehots[:,i,:].to(torch.float))
            nextprob -= torch.logsumexp(nextprob, dim=(1,))[:,None]

            # transition to next hidden state
            totsum = torch.logsumexp(nextprob, dim=(1,)) - torch.log(torch.tensor(n))

            # recombination rate to next SNP
            rprob = torch.exp(self.recomb_prob[i+1])
            no_recomb = (1 - rprob) * torch.exp(nextprob)
            with_recomb = rprob * torch.exp(totsum)
            
            
            no_recomblog = torch.log(1 - rprob) + nextprob
            with_recomblog = torch.log(rprob) + totsum
            # nextprob_arg = torch.logsumexp(torch.log(no_recomb), torch.lo

            # either recombination or no recombination - only two possibilities for haplotype
            nextprob_arg = no_recomb + with_recomb[:,None]
            # nextprob_arg_log = torch.logaddexp(no_recomblog, with_recomblog[:,None])
            
            nextprob = torch.log(nextprob_arg)
            nextprob -= torch.logsumexp(nextprob, dim=(1,))[:,None]
            
            # update the forward probabilities
            fprobx_haplo[:,i+1,:] = nextprob
        self.fprobx_haplo = fprobx_haplo
        return p_hmm
        
    
        
    def forward_backward_alg(self, log_phi_haplos, x_onehots, forward_only=False, z_only=False, training=True):
        # print("running fwd-bwd")
        # ref tensor should be m x n
        # m = nsnps, n = ref panel size
        m = int(self.ref_tensor.shape[0])
        n = int(self.ref_tensor.shape[1])
        
        assert int(log_phi_haplos.shape[1]) == m
        batch_size = int(log_phi_haplos.shape[0])
        
        log_emission_dosage_haplo = torch.log(torch.tensor(self.emission_dosage_haplo)).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # backward
        # set up prevprob, nextprob matrices
        prevprob = torch.zeros(batch_size, n)
        nextprob = torch.zeros(batch_size, n)
        p_xe = torch.zeros(batch_size)

        for i in range(m)[::-1]:
            #print(i, "SNPS completed (backward)")
            # if i % 5000 == 0:
            #     print(i, "SNPS completed")
            # get reference panel for this snp
            ref_dosage_haplo = (self.ref_tensor[i][None,:]).repeat(batch_size, 1)

            # set up matrix to store emission matrix
            state_emission_haplo = torch.zeros(batch_size, n, 2)

            # for both dosages, fill emission matrix
            for dos in range(2):
                emissions = (ref_dosage_haplo == dos)[:,:,None] * log_emission_dosage_haplo[:,dos,:][:,None,:]
                state_emission_haplo += emissions

            # first SNP has no prevprob, but others do
            if i == m-1:
                dosage_prob_arg = self.fprobx_haplo[:,i,:][:,:,None] + state_emission_haplo
                dosage_prob = torch.logsumexp(dosage_prob_arg, dim=(1,))
            else:
                dosage_prob_arg = self.fprobx_haplo[:,i,:][:,:,None] + prevprob[:,:,None] + state_emission_haplo
                dosage_prob = torch.logsumexp(dosage_prob_arg, dim=(1,))

            # many variables introduced here because otherwise pytorch is not pushing gradients in backprop
            dosage_prob_phi = dosage_prob + log_phi_haplos[:,i,:]
            
            # sometimes for numerical stability clamping is necessary, not sure
            # torch.clamp(dosage_prob_phi, -1e-8)
            dosage_prob_scale = torch.logsumexp(dosage_prob_phi, dim=(1,))[:,None]
            dosage_prob_new = dosage_prob_phi - dosage_prob_scale

            # update P(X|E) on the fly
            p_xe_update = torch.sum(dosage_prob_new * x_onehots[:,i,:], dim = 1)
            p_xe += p_xe_update

            # update nextprob
            nextprob_new =  torch.logsumexp(state_emission_haplo + log_phi_haplos[:,i,:][:,None,:], dim=2) 
            #nextprob_new = nextprob_new.clamp(1e-8)
            scaling_factor = torch.logsumexp(nextprob_new, dim=(1,))[:,None]
            nextprob += (nextprob_new - scaling_factor)

            if i == 0:
                break

            # recombination and transition
            totsum = torch.logsumexp(nextprob, dim=(1,)) - torch.log(torch.tensor(n))
            rprob = torch.exp(self.recomb_prob[i])
            no_recomb = (1 - rprob) * torch.exp(nextprob) 
            with_recomb = rprob * torch.exp(totsum)

            # transition
            nextprob_arg = torch.log(no_recomb + with_recomb[:,None])
            nextprob = nextprob_arg - torch.logsumexp(nextprob_arg, dim=(1,))[:, None]

            # set prevprob to nextprob
            prevprob = nextprob 

        log_phi_tot = torch.sum(log_phi_haplos * x_onehots, dim=(1,2))
        if training:
            return loss_fn(p_xe)
        return p_xe

    
    def forward(self, gexp: torch.Tensor, x_onehots: torch.Tensor, forward_only=False, z_only=False, phi_only=False, reid=False, training = True, freq_baseline=True) -> torch.Tensor:
        
        prediction = self.predictor(gexp)
        sig = torch.nn.Sigmoid()
        one_probs = sig(prediction)
        
        zero_probs = 1-one_probs
        prior_probs = torch.stack((zero_probs, one_probs), dim=2)
        expand_probs = torch.zeros(2*prior_probs.shape[0], prior_probs.shape[1], prior_probs.shape[2])
        expand_probs[::2, :, :] = prior_probs   # Index every second row, starting from 0
        expand_probs[1::2, :, :] = prior_probs
        
        if z_only:
            log_phi = torch.log(expand_probs)
            log_phi_tot = torch.sum(log_phi * x_onehots, dim=(1,2))
            return(log_phi_tot)
            
        if training:
            out = self.forward_backward_alg(torch.log(expand_probs), x_onehots)
        
        else:
            #print(torch.log(expand_probs))
            out = self.forward_backward_alg(torch.log(expand_probs), x_onehots, training=False)
        return out



# Single Layer Ordinal Logistic Model
class sparselinear_layer(torch.nn.Module):
    """Single sparse linear layer to map from G genes to S SNPs according to gene-SNP connectivity matrix"""
    def __init__(self, n_genes, n_SNPs, connectivity, bias=True):
        super().__init__()
        self.n_genes = n_genes
        self.n_SNPs = n_SNPs
        self.index = connectivity
        # Set up sparse weight tensor (with connection indices defined by connectivity matrix) -> dim = SxG
        # W_sparse = torch.sparse_coo_tensor(connectivity, torch.zeros(connectivity.shape[1]), size=(n_SNPs, n_genes))
        sparse_vals = torch.Tensor(connectivity.shape[1])
        if bias:
            b = torch.Tensor(n_SNPs,1)
        self.W_sparse = torch.nn.Parameter(sparse_vals)
        self.b = torch.nn.Parameter(b)
        # Initialize weights to zero
        torch.nn.init.zeros_(self.W_sparse)
        torch.nn.init.zeros_(self.b)
        
        
        
    def forward(self, x):
        # X should be a dense tensor of dimension n_samples x n_genes (NxG) -> transpose to GxN
        dot_prod = spmm(index=self.index, value=self.W_sparse, m=self.n_SNPs, n=self.n_genes, matrix=x.t())
        output = dot_prod + self.b
        # output dim -> SxN -> transpose to NxS
        return output.t()

        
def open_and_train(ref_panel, gene_exp, haplos, connectivity_mat, window, size, epochs, recomb, testing = False, ref_size=1000, learning_rate=0.025, device='cpu', train_path=None):
    print("training...")
    # nsnps x nindivs
    ref = np.load(ref_panel)[window*size:(window+1)*size,0:ref_size]
    assert(ref.shape[1] == ref_size)
    
    # nindivs x ngenes
    exp = np.load(gene_exp)
    
    # nindivs x nsnps x 2
    haplos = np.load(haplos)
    haplos[haplos < 0] = 0
    
    haplos = torch.nn.functional.one_hot(torch.tensor(haplos).to(device).long(), num_classes=2)
    
    # indices to connect
    connectivity = np.load(connectivity_mat)
    geno_info = pd.read_csv(recomb, sep="\t")
    
    recomb_probs = torch.tensor(np.array(list(geno_info['recombination_neffective500']))).to(device)

    nsnps = ref.shape[0]
    ngenes = exp.shape[1]
    connectivity_ub = connectivity[:,connectivity[0,:] < (window+1)*size]
    connectivity_lb = connectivity_ub[:,connectivity_ub[0,:] >= window*size]
    nsnps = size
    connectivity_lb[0,:] = connectivity_lb[0,:] - window*size
    
    
    predictor_layer = torch.nn.Sequential(sparselinear_layer(ngenes, nsnps, connectivity=torch.tensor(connectivity_lb).to(device), bias=True))
    
    ref_tensor = torch.tensor(ref).to(device)
    
    model = OrdinalHMM(predictor_layer, ref_tensor, recomb_probs)
    PATH = train_path
    print("files ready...") 
    
    # if not matching:
    gexp = exp #[:530,:]
    gexp_val = exp #[530:588,:]
    print(gexp.shape)
    x_onehots = torch.transpose(haplos, 0, 1)[:,window*size:(window+1)*size,:]


    start = time.time()

    torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    p_hmm = model.forward_pass_hmm(x_onehots)

    outs = []
    vals = []
    for i in range(epochs):
        optimizer.zero_grad()
        out = model.forward(torch.Tensor(gexp), x_onehots)
        # print(out)
        outs.append(float(out.cpu().detach().numpy()))

        out.backward()
        optimizer.step()

    print("Saving model...")
    torch.save(model.state_dict(), train_path)
            
            
    
    
    
    
def match_score(refpanel, haplos, gene_exp, connectivity_mat, window, window_size, ref_size=None, learning_rate=0.025, device='cpu', cache_hmm=True, hmm_path=None, test_path=None):
    
    ref = np.load(refpanel)[window*window_size:(window+1)*window_size]
    if ref_size is not None:
        ref =ref[:,0:ref_size]
        assert(ref.shape[1] == ref_size)
    
    # nindivs x ngenes
    # load from desired test gene exp file
    gexp = np.load(gene_exp)
    
    # nindivs x nsnps x 2
    # load from desired test haplotype file
    haplos = np.load(haplos)
    haplos[haplos < 0] = 0
    
    haplos = torch.nn.functional.one_hot(torch.tensor(haplos).to(device).long(), num_classes=2)
    x_onehots = torch.transpose(haplos, 0, 1)[0:2*gexp.shape[0],window*window_size:(window+1)*window_size,:]
    
    # indices to connect - should be same as training
    connectivity = np.load(connectivity_mat)
    geno_info = pd.read_csv(recomb, sep="\t")
    recomb_probs = torch.tensor(np.array(list(geno_info['recombination_neffective500']))).to(device)
    
    
    nsnps = ref.shape[0]
    ngenes = gexp.shape[1]
    connectivity_ub = connectivity[:,connectivity[0,:] < (window+1)*window_size]
    connectivity_lb = connectivity_ub[:,connectivity_ub[0,:] >= window*window_size]
    nsnps = window_size
    connectivity_lb[0,:] = connectivity_lb[0,:] - window*window_size
    
    # initialize model, load state dict
    predictor_layer = torch.nn.Sequential(sparselinear_layer(ngenes, nsnps, connectivity=torch.tensor(connectivity_lb).to(device), bias=True))
    
    ref_tensor = torch.tensor(ref).to(device)
    
    model = OrdinalHMM(predictor_layer, ref_tensor, recomb_probs)
    PATH = test_path
    
    model.load_state_dict(torch.load(PATH))
    
    if cache_hmm:
        p_hmm = model.forward_pass_hmm(x_onehots)
        np.save(hmm_path, p_hmm.cpu().detach().numpy())
        
    else:
        p_hmm = np.load(hmm_path)
        
    score = model.forward(torch.Tensor(gexp), x_onehots, training=False)
    return p_hmm, score
        
    

def matching(scores, nexamples):
    # organize match scores such that 0th index of 0th axis contains all the correct matches
    # works with dsm, gnb, ebl
    matches = len(np.where(scores[0:nexamples,:].argmax(axis=0) == 0)[0])
    return matches

    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', help='window indexing')
    parser.add_argument('--epochs', help='number of training epochs')
    parser.add_argument('--scoring', help='scoring mode')
    parser.add_argument('--matching', help='matching mode')
    parser.add_argument('--ref_size', help='reference panel size')
    parser.add_argument('--window_size', help='size of window')
    parser.add_argument('--chrom', help='chromosome number')
    parser.add_argument('--learning_rate', help='learning rate')
    parser.add_argument('--testing_haplos', help='path to test set of haplotypes')
    parser.add_argument('--training_haplos', help='path to training set of haplotypes')
    parser.add_argument('--testing_exp', help='path to test set of expression')
    parser.add_argument('--training_exp', help='path to training set of expression')
    parser.add_argument('--recomb', help='path to recombination rates')
    parser.add_argument('--connectivity', help='path to connectivity matrix')
    parser.add_argument('--ref_panel', help='path to reference panel')
    parser.add_argument('--match_path', help='path to save match scores')
    parser.add_argument('--train_path', help="path to save training state")
    parser.add_argument('--scores_path', help="path to score matrix")
    parser.add_argument('--nexamples', help="number of haplos to link")
    parser.add_argument('--cache_hmm', help="cache hmm scores")
    parser.add_argument('--test_path', help="path to .pt file of trained model")
    parser.add_argument('--hmm_path', help="path to save HMM probabilities")

    args = parser.parse_args()
    
    epochs = 50
    if args.epochs is not None:
        epochs = int(args.epochs)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    chrom = 20
    if args.chrom:
        chrom = args.chrom
    
    eqtls = args.training_haplos
    refpanel = args.ref_panel
    connectivity = args.connectivity
    geneexp = args.training_exp
    recomb = args.recomb
    
    
    #window = int(args.window)
    # print("caching version....")
    
    lr = 0.025
    if args.learning_rate:
        lr = float(args.learning_rate)
    
    window = 3
    if args.window:
        window = int(args.window)
            
    ref_size = None
    if args.ref_size:
        ref_size = int(args.ref_size)
    
    window_size = 500
    if args.window_size:
        # print("window size")
        window_size = int(args.window_size)
        
    
    if args.scoring:
        testing = True
        haplos = args.testing_haplos
        # haplos = np.load(eqtls)
        # neqtls = haplos.shape[0]
        # nwindows = int(neqtls / window_size) + 1
        exp = args.testing_exp
        # exp = np.load(gene_exp)
        cache_hmm=None
        hmm_path=None
        if args.cache_hmm is not None:
            cache_hmm = True
            hmm_path = args.hmm_path
        test_path = args.test_path

        
        # for window in range(nwindows):
        with torch.no_grad():
            scores = match_score(refpanel, haplos, exp, connectivity, window, window_size, ref_size=1000, learning_rate=0.025, device='cpu', cache_hmm=cache_hmm, hmm_path=hmm_path, test_path=test_path)
            np.save(args.match_path, scores[1].cpu().detach().numpy())
        sys.exit()
       
    if args.matching:
        scores = np.load(args.scores_path)
        matches = matching(scores, args.nexamples)
        print("Total matches: ", matches)
        #return matches
        sys.exit()
                
        
    else:
        if args.train_path is None:
            raise Exception("Please provide a path to save the trained model")
        
        open_and_train(refpanel, geneexp, eqtls, connectivity, window, window_size, epochs, recomb, testing=False, ref_size=ref_size, learning_rate = lr, device=device, train_path = args.train_path)
        sys.exit()



        
        
        
    
# def randomIndividuals(nindivs=10, ngenes = 4, nsnps = 5, ref_size=1000):
#     nhidden_states=100
#     #gexp = np.random.rand(nindivs, ngenes)
#     gexp_one = np.random.rand(ngenes)
#     gexp = np.repeat(gexp_one[np.newaxis, :], nindivs, axis=0)
    
#     genos = np.random.choice([0,1], size=(nindivs, nsnps), p=[0.5, 0.5])
#     ref = np.random.randint(2, size=(nsnps, ref_size))
#     ref_tensor = torch.Tensor(ref)
#     recombination_probs = torch.tensor(np.random.uniform(low=0, high=0.5, size=(nsnps+1,)))

#     predictor = torch.nn.Linear(ngenes, nsnps, bias=True)

#     model = OrdinalHMM(predictor, ref_tensor, recombination_probs)
#     xs = np.random.randint(2, size=(nsnps,nindivs))
#     x_onehots = np.stack((xs==0, xs==1)).astype(np.int8).transpose()
#     x_onehots = torch.tensor(x_onehots)
#     print(x_onehots.shape)
    
#     torch.autograd.set_detect_anomaly(True)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#     for i in range(20):
#         out = model.forward(torch.Tensor(gexp), x_onehots)
#         print(out)
#         out.backward()
#         optimizer.step()

# if testing:
#         # print("testing....")
#         gexp = exp[test:,:]
#         gexp_copy = exp[test:,:]
#         x_onehots = torch.transpose(haplos, 0, 1)[test*2:,window*size:(window+1)*size,:]
