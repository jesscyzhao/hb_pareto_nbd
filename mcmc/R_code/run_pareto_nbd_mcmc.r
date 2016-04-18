source('three-model-comparison/cz_pareto_mcmc.r')
source('BTYDplus-master/R/pareto-ggg-mcmc.r')
# library(BTYDplus)

params <- list(k=1, r=0.9, alpha=10, s=0.8, beta=12)
n <- 5000
cbs <- pggg.GenerateData(n, 52, 52, params, TRUE)$cbs

init_param = NULL
num_iter = 5000
thin = 50
output = cz_mcmc_multi_core(cbs, num_iter = 5000, burn_in = 1000, thin = 50)


library(coda)
mcmc_sample_list = lapply(1:length(output$gamma_param), function(i) as.mcmc(output$gamma_param[[i]], burnin, thin))

hist(mcmc_sample_list[[1]][, 'r'], 100, main='Distribution of r')
hist(mcmc_sample_list[[1]][, 'alpha'], 100, main='Distribution of alpha')
hist(mcmc_sample_list[[1]][, 's'], 100, main='Distribution of s')
hist(mcmc_sample_list[[1]][, 'beta'], 100, main='Distribution of beta')

autocorr.plot(mcmc_sample_list[[1]])
traceplot(mcmc_sample_list[[1]])
param_names = dimnames(mcmc_sample_list[[1]])[[2]]
for (i in 1:length(param_names)){
  plot(calc.running.mean(mcmc_sample_list[[1]][, param_names[i]])~seq(1, num_iter/thin), main=paste('Running mean ', param_names[i]) , type='l', xlab='# of draws', ylab='Running mean')
}

lapply(mcmc_sample_list, summary)
gelman.diag(mcmc_sample_list)
