rm(list = ls())

library(reticulate)
library(MASS)
np <- import("numpy")
# constants
n_train_day = 1  # number of training days
n_outcome = 40  # number of outcome measurements
action_ids = "12"  # action patient ids
# outcome_ids = "12,28,29" # outcome patient ids
# args
args = commandArgs(trailingOnly=TRUE)
sample_path = args[1]
period = args[2]
n_patient = args[3]  # number of patients
outcome_ids = args[4]
seed = args[5]
#

run_folder <- paste("/np", n_patient, ".ntr", n_train_day, ".pa", action_ids, ".po", outcome_ids, ".no", n_outcome, ".s", seed, sep='')
period_folder <- paste("/", period, sep='')
samples_folder <- paste(sample_path, run_folder, period_folder, sep='')
npz <- np$load(paste(samples_folder, "/outcome_r_data_fpca.npz", sep=''))

Y_data <- data.frame(ID=npz$f[['ID']], Time=npz$f[['Time']], Treatment=npz$f[['Treatment']],
                Outcome=npz$f[['Outcome']], Mediator=npz$f[['Mediator']])

# Y_data$Treatment[601:1200] <- 1
# print(Y_data$Time)
work_grid=seq(0,1,length=40)
Y_data <- Y_data[1:nrow(Y_data)-1,]
source("models/benchmarks/fpca/R_script/simulation_outcome_fpca.R")

np$savez(paste(samples_folder, "/fpca_mcmc_output.npz", sep=''), eigen=eigen, beta=beta_coeff, aug_x=AUG_X, theta=theta)
