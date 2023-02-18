library("peer", lib.loc="~/local/R/library")
library("optparse")
 
option_list = list(
  make_option(c("-f", "--expression_file"), type="character", default=NULL, 
              help="merged expression data file name", metavar="character"),
    make_option(c("-o", "--out"), type="character", default="out.txt", 
              help="output file name [default= %default]", metavar="character"),
    make_option(c("-tr", "--train"), type="integer", default=588, 
              help="number of training examples")
    make_option(c("-te", "--test"), type="integer", default=292, 
              help="number of testing examples")
    
); 
 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

# Load expression TPM file
exp_data <- read.csv(opt$expression_file)

# Process data (drop non-numeric columns, convert to matrix, take transpose)
# drop <- (any non-numeric columns)
exp_data_filt <- exp_data[,!(names(exp_data) %in% drop)]
exp_data_filt <- t(as.matrix(exp_data_filt))

# Set covariate vector (558 GTEx samples)
cov_vec <- c(rep(0,opt$train), rep(1,opt$test))
cov_vec <- as.matrix(cov_vec)

# Set up PEER model
model=PEER()
PEER_setNk(model, 50)
PEER_setPhenoMean(model, exp_data_filt)
PEER_setCovariates(model, cov_vec)

# Run model
PEER_update(model)

# Obtain PEER residuals
PEER_residuals <- PEER_getResiduals(model)
