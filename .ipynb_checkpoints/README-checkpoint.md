# DSM
Code for paper "Assessing transcriptomic re-identification risks using discriminative sequence models" accepted for oral presentation at RECOMB 2023.


## Training DSM
DSM is trained separately for each window of eQTLs. We use a modified version of the forward-backward algorithm for HMM inference which is described fully in our paper.

The primary arguments required for training are:
* window: an index of the window you want to train
* window_size: size of the window you want to train. The window will go from window * window_size to (window + 1) * window_size
* connectivity: a matrix specifying the connections between indices of the gene expression vector and genotype vector. It should be a 2 x n_connections sized numpy matrix, where the first row indexes into the gene expression vector and second into the genotype vector. connections should be prebuilt based on the eqtl set defined.
* training_exp: path to a numpy matrix containing training gene expression vectors. It should be nindividuals x ngenes.
* training_haplos: path to a numpy matrix containing the training haplotypes. It should nindivs x nsnps x 2. The x2 is due to each individual carrying two haplotypes.
* ref_panel: path to a numpy matrix containing the reference panel haplotypes. It should be nsnps x nrefindivs. Optionally you can provide the ref_size argument to subset the reference panel. The reference panel should be chosen carefully. Larger reference panels provide better and more stable training.
* epochs: number of epochs to train
* learning_rate: learning rate for Adam optimizer
* train_path: path to save the trained model
* recomb: a pandas dataframe containing SNP coordinates for all eQTLs considered, along with the recombination rates. The field recombination_neffective500 is retrieved as the recombination rates. Example recombination rate generation is provided in recombination.py.

Example data formats are provided in the example_data folder. DSM can be run with python DSM.py --argument1 [ARGUMENT 1] --argument2 [ARGUMENT 2]


## Scoring
Once trained, you can use DSM to score matches between example gene expression vectors and genotype vectors. The primary arguments required are:
* window: an index of the window you want to score
* window_size: size of the window you want to score. The window will range from window * window_size to (window + 1) * window_size
* connectivity: a matrix specifying the connections between indices of the gene expression vector and genotype vector. It should be a 2 x n_connections sized numpy matrix, where the first row indexes into the gene expression vector and second into the genotype vector. connections should be prebuilt based on the eqtl set defined.
* testing_exp: path to a numpy matrix containing gene expression vectors to score. It should be nindividuals x ngenes.
* testing_haplos: path to a numpy matrix containing the haplotypes to score. It should nindivs x nsnps x 2. The x2 is due to each individual carrying two haplotypes.
* ref_panel: path to a numpy matrix containing the reference panel haplotypes. It should be nsnps x nrefindivs. Optionally you can provide the ref_size argument to subset the reference panel. Note that the **same** reference panel should be used in both testing and training to ensure optimal stability.
* test_path: path to a saved PyTorch model
* recomb: a pandas dataframe containing SNP coordinates for all eQTLs considered, along with the recombination rates. The field recombination_neffective500 is retrieved as the recombination rates. Example recombination rate generation is provided in recombination.py.
* matching: turns on scoring mode
* scores_path: path to save the scores for the (gene expresion, genotype) pair passed above. Returns both the hmm score of the probability of the genotype alone as well as the DSM score for the match between the gene expression and genotype pair

It is important that the reference panel used to train and the reference panel used to score are the same. Similarly, it also important that the SNP set is the same. Shifts in these may result in unstable or inaccurate scores. You can optionally call cache_hmm and/or hmm_path to cache and save the results of just the genotype probabilities. Once you have retrieved the scores, you can concatenate these scores into a matrix. The 0th index of the 0th axis should contain all the correct scores (of true matches), while the remaining indices should contain incorrect scores. The matching() function can find the links, for any of the three methods (DSM, EBL, GNB). 

Code to set the recombination rates can also be found in the models folder.

## GNB and EBL
We have implemented both GNB and EBL. As described in our paper, our implementation of EBL uses a hybrid method which defaults to the scores of GNB for non-extreme SNPs. The arguments to GNB/EBL are similar to DSM, however, some differences:
* train_test_split: Index to split the training and testing data. Instead of passing separate training/testing matrices, GNB/EBL are able to handle the two at once. This is due to their negligible memory requirements relative to DSM.
* ebl_path/gnb_path: Path to save outputs of these methods
* genos: Both of these methods use genotypes instead of haplotypes, so please pass in a genotype matrix, with the same dimensions nindivs x nsnps (no x2).
* **EBL only** corr: Correlation threshold between gene expression and genotypes. Above this (absolute) correlation threshold, SNPs are considered "extreme enough" for EBL.
* **EBL only** extremity: Extremity threshold for predicting a genotype. If the test gene expression is above the chosen extremity threshold in the training distribution of that gene's expression, then we will use the EBL model.

## Data Cleaning
Data is cleaned with VCFTools, BCFTools, Plink2, PEER, LiftOver, and Tabix. To subset a VCF based on a predefined list of samples, use [this](https://www.biostars.org/p/184950/). SNPs are aligned between different builds with LiftOver. All VCFs are converted first into .haps files and then converted to numpy matrices using the haps_to_numpy.py file in the cleaning folder. An example of converting VCF to .haps using plink2 is provided [here](https://www.biostars.org/p/292843/). Example PEER factor normalization is provided in the cleaning folder as well.
Additional example data cleaning scripts are provided in the cleaning folder. 

