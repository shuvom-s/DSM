# Discriminative Sequence Model (DSM)
This repository provides code for this paper:

S. Sadhuka, D. Fridman, B. Berger, and H. Cho _Assessing transcriptomic re-identification risks using discriminative sequence models_,  [RECOMB 2023](http://recomb2023.bilkent.edu.tr/program.html). 

The paper is also under review at _Genome Research_.

## Dependencies
Our code to run the DSM is primarily written in Python. Most dependencies (e.g. `numpy`, `pandas`) are available in any standard conda installation. The remaining dependencies that do not ship with conda are:
```
torch >= 1.8.1
torch_sparse >= 0.6.11
``` 


## Training

### Overview
DSM is trained separately for each window of eQTLs with a panel of reference haplotypes to learn correlation patterns between SNPs. For reference, a window of 750 eQTLs with a reference panel of 1000 haplotypes requires around 7 hours and 75GB of memory to train per window. Larger reference panels and window sizes are recommended for better and more stable results. We use a modified version of the forward-backward algorithm for forward pass inference which is described fully in our paper.

### Input Files and Arguments
Training DSM requires several input files to train, all of which are `numpy` matrices unless otherwise specified. These files are:
* _Gene expression matrix_: a matrix with dimensions [# training individuals x # genes] containing the gene expression profiles for each training individual.
* _Haplotype matrix_: a matrix with dimensions [# training individuals x # SNPs (eQTLs) x 2] containing phased haplotypes for each training individual. It is important that the order of individuals present in the haplotype matrix is the same as the order in the gene expression matrix.
* _Reference panel_: a matrix with dimensions [# SNPs x # reference panel individuals] containing phased haplotypes from a reference panel. Importantly, any overlapping haplotypes between the reference panel and training/testing haplotypes should be removed. Larger reference panels are strongly recommended.
* _Connectivity_: a matrix with dimensions [2 x # eGene-eQTL pairs] specifying the graph structure to connect eQTLs to eGenes. For instance, (2,3) would connect the second eGene to the third eQTL in the hapltype vector.
* _Recombinations_: a `.tsv` file read into `pandas` containing the recombination rates for pairs of adjacent SNPs. This should also provide the coordinates for each SNP in a standard genetic map (e.g. hg38).

There is only one output file that is generated during training:
* _Model_: a `.pt` file caching the parameters of the trained PyTorch model.

The above input files are passed as arguments to `DSM.py`. In general, arguments are specified with the `--argument` flag. For instance, to specify the learning rate one can use `--learning_rate 0.001` or equivalent. Specific argument flags are explained below.
* `--training_exp`: path to a numpy matrix (`.npy` file) containing training gene expression vectors. As noted above, it should have dimensions [# training individuals x # genes].
* `--training_haplos`: path to a numpy matrix (`.npy` file) containing training haplotypes (phased). As noted above, it should have dimensions [# training individuals x # SNPs (eQTLs) x 2].
* `--ref_panel`: path to a numpy matrix containing the reference haplotypes. As noted above, it should have dimensions [# SNPs x # reference panel individuals].
* `--connecitivty`: path to a numpy matrix speicfying the connections between indices of the gene expression vector and genotype vector. It should be [2 x # eGene-eQTL pairs]. The first row should index into the gene expression vector and the second row into the haplotype vector. As noted above, (2,3) would connect the second eGene to the third eQTL in the hapltype vector. The connectivity matrix should be prebuilt based on the eQTL set and eGene set. Lower p-value thresholds will result in a sparser edge set.
* `--window`: an index of the window you want to train. The default is 500, but larger window sizes are recommended if sufficient compute is available
* `--window_size`: size of the window you want to train. The window will range from window * window_size to (window + 1) * window_size along the eQTLs dimension. So, if there are 10,000 eQTLs and `--window 1 --window_size 1000` is specified, then the window will range from eQTL 1000 to eQTL 2000.
* `--ref_size`: the number of reference haplotypes to use. The reference panel will be subsetted to only include the first `ref_size` haplotypes.
* `--epochs`: number of epochs to train DSM. One epoch corresponds to one pass over all individuals in the training dataset. The default is 50 epochs, but the optimal number of epochs can vary from window to window or training set to set. We recommend setting epochs using cross-validation.
* `--learning_rate`: learning rate for Adam optimizer. The default rate is 0.025, but the optimal number of epochs can vary from window to window or training set to set. We recommend setting the learning rate using cross-validation.
* `--train_path`: path to save the trained model
* `--recomb`: a `pandas` dataframe (can be saved as `.tsv` file, but read in through `pandas`) containing SNP coordinates for all eQTLs considered, along with the recombination rates. The field recombination_neffective500 is retrieved as the recombination rates. Example recombination rate generation is provided in `recombination.py`.

Please note that the training is sensitive to the reference panel. Substantial mismatches between the genetic architectures in the training set and the reference panel can lead to poor training.

Example data formats are provided in the `examples/` folder along with an example training script `example.sh`. These files contain synthetic data  — and hence should not be interpreted — but instead provide the proper dimensions for all of the matrices specified above.


## Scoring
Once trained, you can use DSM to score matches between example gene expression vectors and genotype vectors. The input file formats are similar to the training step. These files are
* _Gene expression matrix_: a matrix with dimensions [# individuals to score x # genes] containing the gene expression profiles for each individual to score.
* _Haplotype matrix_: a matrix with dimensions [# individuals to score x # SNPs (eQTLs) x 2] containing phased haplotypes for each individual to score. It is important that the order of individuals present in the haplotype matrix is the same as the order in the gene expression matrix.
* _Reference panel_: a matrix with dimensions [# SNPs x # reference panel individuals] containing phased haplotypes from a reference panel. Importantly, any overlapping haplotypes between the reference panel and training/testing haplotypes should be removed. Larger reference panels are strongly recommended. It is crucial that the reference panel used to score is the same as the one used to train.
* _Connectivity_: a matrix with dimensions [2 x # eGene-eQTL pairs] specifying the graph structure to connect eQTLs to eGenes. For instance, (2,3) would connect the second eGene to the third eQTL in the hapltype vector. The connectivity matrix should be reused from training time. A different connectivity structure (e.g. different tissue) or SNP set might yield unstable results.
* _Model_: a path to trained DSM model with cached parameters.
* _Recombinations_: a `.tsv` file read into `pandas` containing the recombination rates for pairs of adjacent SNPs. This should also provide the coordinates for each SNP in a standard genetic map (e.g. hg38). The recombination file should be reused from training time.

There are several possible outputs during scoring:
* _Scores_: a `.npy` file containing a vector (or matrix) of scores between genotype vectors and gene expression vectors.
* _HMMs_: a `.npy` file containing a vector (or matrix) of probabilities for only the genotype vectors. These probabilities are outputs of the standard Li-Stephens HMM.

The primary arguments required are:
* `--test_path`: path to a `.pt` file containing cached parameters for a trained DSM model.
* `--training_exp`: path to a numpy matrix (`.npy` file) containing gene expression vectors to score. As noted above, it should have dimensions [# individuals to score x # genes].
* `--training_haplos`: path to a numpy matrix (`.npy` file) containing haplotypes (phased) to score. As noted above, it should have dimensions [# individuals to score x # SNPs (eQTLs) x 2].
* `--ref_panel`: path to a numpy matrix containing the reference haplotypes. As noted above, it should have dimensions [# SNPs x # reference panel individuals].
* `--connecitivty`: path to a numpy matrix speicfying the connections between indices of the gene expression vector and genotype vector. It should be [2 x # eGene-eQTL pairs]. It should be reused from training time.
* `--window`: an index of the window you want to score. It should be correspond to the window used for the model in `test_path`.
* `--window_size`: size of the window you want to score. It should be correspond to the window size used for the model in `test_path`.
* `--ref_size`: the number of reference haplotypes to use. The reference panel will be subsetted to only include the first `ref_size` haplotypes.
* `--match_path`: path to a `.npy` file where the match scores will be saved.
* `--recomb`: a `pandas` dataframe (can be saved as `.tsv` file, but read in through `pandas`) containing SNP coordinates for all eQTLs considered, along with the recombination rates. The field recombination_neffective500 is retrieved as the recombination rates.

It is important that the reference panel used to train and the reference panel used to score are the same. Similarly, it is also important that the SNP set and tissue are the same. Shifts in these may result in unstable or inaccurate scores. 

You can optionally call `--cache_hmm True` `--hmm_path path_to_save.npy` to cache and save the results of just the haplotype probabilities (i.e. the Li-Stephens probabilities for the haplotypes you are scoring). The output is saved in the path specified by the `--hmm_path` argument. Once you have retrieved the scores, you can concatenate these scores into a matrix using `np.concatenate((correct_scores, [list of incorrect_scores]))` or equivalent. The 0th index of the 0th axis should contain all the correct scores (of true matches), while the remaining indices should contain incorrect scores for the same individual. The `matching()` function in `DSM.py` can find the links, for any of the three methods (DSM, EBL, GNB). The function must be called on a concatenated matrix of scores (concatenated as above) and returns the number of rows for which the `max` score is present in the 0th index. 


## GNB and EBL
We have implemented methods corresponding to the following two papers:
* E. Schadt, S. Woo, and K. Hao. _Bayesian method to predict individual SNP genotypes from gene expression data_. _Nature Genetics_ (2012).
* A. Harmanci and M. Gerstein. _Quantification of private information leakage from phenotype-genotype data: linking attacks_. _Nature Methods_ (2016).

We refer to these methods as Gaussian Naive Bayes (GNB) and Extremity-based Linking (EBL) respectively. Full descriptions of these methods are available in the citations above. As described in our paper, our implementation of EBL uses a hybrid method which defaults to the scores of GNB for non-extreme SNPs. The arguments to GNB/EBL are similar to DSM. Unlike DSM, training and testing are done in one call. This is due to the substantially smaller memory and time requirements for training GNB/EBL compared to DSM. Some differences in arguments include:
* _Gene expression matrix_ (`--gene_exp`) is now just a single file that contains both the training and testing data. The matrix should a `.npy` file that is of dimensions [# training + scoring individuals x # genes].
* _Genotype matrix_ (`--genos`) is now a single `.npy` file of dimensions [# SNPs x # training + scoring individuals]. The genotypes should be _unphased_.
* `--train_test_split`: Index to split the training and testing data. As noted above, instead of passing separate training/testing matrices, GNB/EBL are able to handle the two at once. This is due to their negligible memory requirements relative to DSM. The `train_test_split` index will split, for instance, the gene expression matrix at index 500 (if `--train_test_index 500` is passed in) where the first 500 individuals will be used for training, and the remaining individuals for scoring.
* `--ebl_path / --gnb_path`: Path to save outputs of these methods. The output will be posterior matrix of probabilities corresponding to probability of observing [0, 1, 2] at each SNP for each individual. It will be of dimension [# SNPs x # individuals x 3].
* **EBL only** `--corr`: Correlation threshold between gene expression and genotypes. Above this (absolute) correlation threshold, SNPs are considered "extreme enough" for EBL. The default is 0.45. 
* **EBL only** `--extremity`: Extremity threshold for predicting a genotype. If the test gene expression is above the chosen extremity threshold in the training distribution of that gene's expression, then we will use the EBL model. The default is 0.95 (so must be at the top or bottom 5% of the distribution of expression). Both the correlation and extremity thresholds 

## Data Preprocessing
We provide data preprocessing scripts. While there are many possible routes to preprocess data, we recommend the following software:
```{python}
VCFTools
Plink2
BCFTools
LiftOver
```
We recommend the following pipeline:
1. Subset the training/testing VCF(s) based on a predefined list of samples (e.g. those samples which have expression). See [this](https://www.biostars.org/p/184950/) for an example with `BCFTools`.
2. Align the training/testing VCFs if there is a genetic map mismatch using LiftOver. See [this](https://genome.ucsc.edu/cgi-bin/hgLiftOver) tool for an example.
3. Phase any unphased genotypes. Phasing can be done with the [Michigan Imputation Server](https://imputationserver.sph.umich.edu/index.html#!).
4. Subset the reference panel haplotypes and training/testing haplotypes based on the desired SNP coordinates. We provide `filterhaps.py` as an example. `filterhaps.py` takes in a list of lines to keep (as a `.txt` file) and a `.haps` file and converts these into a `.npy` matrix. VCFs can be converted to `.haps` files like [this](https://www.biostars.org/p/292843/).
5. Align gene expression data and normalize it with `PEER`. An example script is provided in `cleaning/PEER_residuals.R`.
6. Generate the connectivity matrix by getting the eGene-eQTL pairs. An example script is provided in `cleaning/GTEx_FUSION_command_line.py`. This file also requires an eQTL info `.tsv` file specified via the `--eqtl_path` argument.
7. Code to set the recombination rates can also be found in the `models/` folder as `recombination.py`. We assume that a `.tsv` file is passed in containing a standard eQTL coordinate set (e.g. `hg38`). You should also pass in a genetic map of the same build as the eQTL set via `--map_path` that contains centiMorgan coordinates. `--outpath` specifies the output path. An example of the input format expected is given in `examples/recombinations.txt.gz` (excluding the columns starting with `recombination_`). The genetic map can be a standard hg38 genetic map file.

You can contact the authors at ssadhuka@mit.edu and dfridman@broadinstitute.org.

