## example training
echo "training"
python ../models/DSM.py  --training_exp synthetic_expression.npy --training_haplos synthetic_eqtls.npy --connectivity synthetic_connectivity.npy --recomb recombinations.txt.gz --ref_panel synthetic_refpanel.npy --learning_rate 0.1 --train_path example.pt --window 0 --window_size 10 --ref_size 10 --epochs 50

echo "scoring"
## example scoring - here scoring data is same as training data (just for demonstration)
python ../models/DSM.py  --testing_exp synthetic_expression.npy --testing_haplos synthetic_eqtls.npy --connectivity synthetic_connectivity.npy --recomb recombinations.txt.gz --ref_panel synthetic_refpanel.npy--test_path example.pt --window 0 --window_size 10 --ref_size 10 --match_path match_true.npy --scoring True --cache_hmm True --hmm_path hmm_probs.npy

## example scoring - here scoring data is same as training data (just for demonstration). in this example, the genotypes and gene expression profiles being matched are NOT true matches
python ../models/DSM.py  --testing_exp synthetic_expression.npy --testing_haplos synthetic_negative_eqtls.npy --connectivity synthetic_connectivity.npy --recomb recombinations.txt.gz --ref_panel synthetic_refpanel.npy --test_path example.pt --window 0 --window_size 10 --ref_size 10 --match_path match_false.npy --scoring True  --cache_hmm True --hmm_path hmm_probs.npy 

## match_true = np.load("match_true.npy")
## match_false = np.load("match_false.npy")
## matches = np.concatenate((match_true, match_false))
## np.save("matches.npy", matches)

## get number of matches
# python ../models/DSM.py  --matching True --scores_path matches.npy

## training and testing GNB
python ../models/GNB.py --gene_exp synthetic_expression.npy --genos synthetic_eqtls_genotypes.npy --train_test_split 500 --gnb_path gnb_probabilities.npy --connectivity synthetic_connectivity.npy
# 
## training and testing EBL
## note this is synthetic data so the correlation threshold is set very low so that some SNPs are actually correlated
python ../models/EBL.py --gene_exp synthetic_expression.npy --genos synthetic_eqtls_genotypes.npy --train_test_split 500 --ebl_path ebl_probabilities.npy --corr 0.01 --extremity 0.9  --connectivity synthetic_connectivity.npy --prob_mat gnb_probabilities.npy