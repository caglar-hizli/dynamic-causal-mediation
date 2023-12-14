# Libraries
# install.packages("rstan")
library("rstan")
options(mc.cores = as.numeric(Sys.getenv('SLURM_CPUS_PER_TASK',8)))
#rstan_options(javascript=FALSE)

# Preparation of data for .stan model
#setwd("../../../../")
task = Sys.getenv('SLURM_ARRAY_TASK_ID')
args = commandArgs(trailingOnly=TRUE)
period = args[1]
deltaT = args[2]

# Training data
N = read.table(paste('./data/stan/', period, '/N.txt', sep=''), sep=' ')[,c(1)]
M = read.table(paste('./data/stan/', period, '/M.txt', sep=''), sep=' ')[,c(1)]
y = read.table(paste('./data/stan/', period, '/y.txt', sep=''), sep=' ')
t = read.table(paste('./data/stan/', period, '/t.txt', sep=''), sep=' ')
x1 = read.table(paste('./data/stan/', period, '/x1.txt', sep=''), sep=' ')[,c(1)]
tx = read.table(paste('./data/stan/', period, '/tx.txt', sep=''), sep=' ')[,c(1)]
hypers = read.table(paste('./data/stan/', period, '/params.txt', sep=''), sep=' ')
trend_p = read.table(paste('./data/stan/', period, '/trend_p.txt', sep=''), sep=' ')[,c(1)]
P = hypers[1,1]
N_max = hypers[2,1]
M_max = hypers[3,1]
PM = hypers[4,1]
Mcumsum = cumsum(M)
Mcumsum = c(0, Mcumsum)

its = 4000

data = list(P=P,N=N,M=M,N_max=N_max,M_max=M_max,PM=PM,y=y,t=t,x1=x1,tx=tx,Mcumsum=Mcumsum,trend_p=trend_p)

# Results Folder
results_folder_base <- './out'
dir.create(results_folder_base, showWarnings = FALSE)
results_folder <- paste(results_folder_base,'/time-correction-',deltaT, sep='')
dir.create(results_folder, showWarnings = FALSE)
period_folder <- paste(results_folder,'/',period, sep='')
dir.create(period_folder, showWarnings = FALSE)
# Run model
fit_obj = stan(file = paste("./time-correction/SimpleModelHier", deltaT, ".stan", sep=""),
               data = data, iter = its, warmup=its/2, chains = 2, cores=8,
               init = list(list(beta1 = 0.07, beta1_p = rep(0.07,P), tx_star = tx, alpha1 = 0.35, alpha1_p = rep(0.35,P), sig_y=rep(0.58,P)),
                           list(beta1 = 0.08, beta1_p = rep(0.08,P), tx_star = tx+0.033, alpha1 = 0.333, alpha1_p = rep(0.333,P), sig_y=rep(0.6,P))),
               pars = c('resp_sum', 'resp_sum1','log_lik','alpha1_p','beta1_p','tx_star'), include=TRUE, save_warmup=FALSE,
               seed = task)


# Save results
# Samples
samples = extract(fit_obj, permuted = TRUE)

# Loo computations
loo1 = loo(fit_obj, pars = "log_lik")
print(loo1)
png(paste(period_folder,'/loo_',task,'.png', sep=''))
plot(loo1)
dev.off()

# Fitting statistics
png(paste(period_folder,'/hist_alpha1_p_',task,'.png', sep=''))
plot(fit_obj, show_density = TRUE, pars = c("alpha1_p"), ci_level = 0.95, fill_color = "purple")
dev.off()

png(paste(period_folder,'/hist_beta1_p_',task,'.png', sep=''))
plot(fit_obj, show_density = TRUE, pars = c("beta1_p"), ci_level = 0.95, fill_color = "blue")
dev.off()

# Convert 3D arrays of predicted ys to 2D for 1st meal/2nd meal/overall
samples_y = data.frame(matrix(0.0,nrow = P,ncol = N_max))
for (p in 1:P) {
  for (n in 1:N[p]) {
    samples_y[p,n] = mean(samples$resp_sum[,p,n])
  }
}
write.csv(samples_y,paste(period_folder,'/samples_y_',task,'.csv', sep=''), row.names = FALSE)

samples_y = data.frame(matrix(0.0,nrow = P,ncol = N_max))
for (p in 1:P) {
  for (n in 1:N[p]) {
    samples_y[p,n] = mean(samples$resp_sum1[,p,n])
  }
}
write.csv(samples_y,paste(period_folder,'/samples_y1_',task,'.csv', sep=''), row.names = FALSE)

# Time corrections
samples_y = data.frame(matrix(0.0,nrow = P,ncol = M_max))
for (p in 1:P) {
  for (m in 1:M[p]) {
    samples_y[p,m] = mean(samples$tx_star[,Mcumsum[p]+m])
  }
}
write.csv(samples_y,paste(period_folder,'/time_corrections.csv', sep=''), row.names = FALSE)

samples_y = data.frame(matrix(0.0,nrow = P,ncol = M_max))
for (p in 1:P) {
  for (m in 1:M[p]) {
    samples_y[p,m] = tx[Mcumsum[p]+m]
  }
}
write.csv(samples_y,paste(period_folder,'/true_times.csv', sep=''), row.names = FALSE)

## Save fitted params for each sample
fitted_params_patients = data.frame(matrix(NA, nrow = P, ncol = 2))

samples_y = numeric(P)
for (p in 1:P) {
  samples_y[p] = median(samples$alpha1_p[,p])
}
fitted_params_patients[1]=samples_y

samples_y = numeric(P)
for (p in 1:P) {
  samples_y[p] = median(samples$beta1_p[,p])
}
fitted_params_patients[2]=samples_y

write.csv(fitted_params_patients,paste(period_folder,'/fitted_params_patients_',task,'.csv', sep=''), row.names = FALSE)
