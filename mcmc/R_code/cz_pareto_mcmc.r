library(parallel)
######################################## Gamma Metroplis Hastings ###########################################

log_gamma_posterior = function(prior, hyper_prior, x){
    shape = prior[1]
    rate = prior[2]
    
    shape_hyper_1 = hyper_prior[1]
    shape_hyper_2 = hyper_prior[2]
    rate_hyper_1 = hyper_prior[3]
    rate_hyper_2 = hyper_prior[4]
    sum_x = sum(x)
    sum_log_x = sum(log(x))
    n = length(x)
    
    log_post = n * (shape * log(rate) - lgamma(shape)) + (shape-1) * sum_log_x - rate * sum_x + 
      (shape_hyper_1 - 1) * log(shape) - (shape * shape_hyper_2) +
      (rate_hyper_1 - 1) * log(rate) - (rate * rate_hyper_2);
    return (log_post)
}  

# mh chain 
mh_gamma_chain = function(gamma_data, init_prior, hyper_prior, burn_in, num_iter, thin, trace=100){
  num_draws = (num_iter-1)%/%thin + 1
  output = array(NA_real_, dim=c(num_draws, 2))
  
  # Initialization 
  current_param  = init_prior
  counter = 0
  for (step in 1:(burn_in + num_iter)){
    
    if (step %% trace==0) cat('step', step, ' of total ', burn_in + num_iter, ' steps \n')
    if (step > burn_in & (step-1-burn_in)%%thin == 0){
      iter = (step-1-burn_in)%/%thin + 1
      output[iter, ] = current_param
    }      
    
    # update step
    one_mh_step_result = mh_gamma_step(gamma_data, current_param[1], current_param[2], hyper_prior)
    
    # minor step to calculate acceptance rate
    if (one_mh_step_result$shape!=current_param[1]){
      counter = counter + 1
    }
    if (one_mh_step_result$rate!=current_param[2]){
      counter = counter + 1
    }
    current_param[1] = one_mh_step_result$shape
    current_param[2] = one_mh_step_result$rate
  } 
  
  return (list(sample_output = output, accept_rate = counter/(2*(burn_in+num_iter))))
}


mh_gamma_step = function(x, shape, rate, hyper_prior){
  
    prior = c(shape, rate)
    lower = 0
    upper = 100
    
    for (i in 1:2){
      propose_prior = prior 
      current_log_post = log_gamma_posterior(prior, hyper_prior, x)
      # proposal distribution is normal 
      # sample within boundary (0, 100)
      propose_param = pmin(pmax(1e-10, prior[i] + rnorm(1)), 100)
      propose_prior[i] = propose_param
      propose_log_post = log_gamma_posterior(propose_prior, hyper_prior, x)
      
      u = runif(1)
      # r = exp(log(propose posterior)-log(current posterior)) = propose_posterior/current_posterior 
      # accept with r, thus sample from uniform, since u < 1, if r > 1, then always accept; if r < 1, P(u < r) = r 
      if (u < exp(propose_log_post-current_log_post)){
        prior = propose_prior 
      }
    }
    
    return(list(shape = prior[1], rate=prior[2]))
}


######################################## Pareto NBD Gibbs sampling ################################################


slice_sampling_step = function(x, shape, rate, hyper_prior, steps=10, lower=1e-5, upper=100, w = 1, quiet = TRUE){
    prior = c(shape, rate)
    sum_x = sum(x)
    sum_log_x = sum(log(x))
    n = length(x)
    log_post_prior = log_gamma_posterior(prior, hyper_prior, x)
    
    
    
    for (i in 1:steps){
      
      if (!quiet) cat('step', i, ' in the chain \n')
      for (j in 1:length(prior)){
          center = prior 
          left = prior 
          right = prior 
          log_post_z = log_post_prior - rexp(1)
          
          u = runif(1)*w
          left[j] = pmax(center[j] - u, lower)
          right[j] = pmin(center[j] + (w-u), upper)
          
          while(left[j]>lower){
             if(log_gamma_posterior(left, hyper_prior, x) > log_post_z){
                 left[j] = left[j] - w
             }
             else{
               break
             }
          }
          while(right[j]<upper){
             if( log_gamma_posterior(right, hyper_prior, x) > log_post_z){
             right[j] = right[j] + w
             }
             else{
               break
             }
          }
          
          param_lower = pmax(lower, left[j])
          param_upper = pmin(upper, right[j])
          
          propose_prior = center          
          count=1
          
          while (count < 1e4){
            
            propose_param = runif(1, param_lower, param_upper)
            propose_prior[j] = propose_param
            
            log_propose_prior = log_gamma_posterior(propose_prior, hyper_prior, x) 
            if (log_propose_prior > log_post_prior){
              break 
            }
            
            if (propose_param < prior[j]){
              param_lower = propose_param
            }
            else{
              param_upper = propose_param
            }
            count = count + 1
          }
          
          prior = propose_prior
          log_post_prior = log_propose_prior
          if(!quiet) print(prior)
      }
      
    }
    
    return (list(shape = prior[1], rate = prior[2]))
  
  
}


draw_gamma_parameters = function(latent_variables, gamma_parameters, all_hyper_priors, choice='slice_sampling'){
  
  sampling_function = if (choice=='slice_sampling') slice_sampling_step else mh_gamma_step
  
  r_alpha_update = sampling_function(latent_variables['lambda', ], gamma_parameters['r'], gamma_parameters['alpha'], all_hyper_priors[1:4])
  
  gamma_parameters['r'] = r_alpha_update$shape
  gamma_parameters['alpha'] = r_alpha_update$rate
  s_beta_update = sampling_function(latent_variables['mu', ], gamma_parameters['s'], gamma_parameters['beta'], all_hyper_priors[5:8])
  
  gamma_parameters['s'] = s_beta_update$shape
  gamma_parameters['beta'] = s_beta_update$rate
  
  return (gamma_parameters)
  
}




draw_latent_variables = function(data, latent_variables, gamma_parameters){
  
    # draw individual lambda
    draw_lambda = function(r, alpha, x, T.cal, tau){
      N      = length(x)
      lambda  = rgamma(n = N, 
                       shape = r + x, 
                       rate = alpha + pmin(T.cal, tau))
      return(lambda)
    }
    
    # draw individual mu 
    draw_mu = function(s, beta, tau, T.cal){
      N = length(T.cal)
      mu = numeric(N)
      alive = tau > T.cal
      
      if (any(alive)){
        mu[alive] = rgamma(n=sum(alive), 
                           shape = s, 
                           rate = beta + T.cal[alive])
      }
      
      if (any(!alive)){
        mu[!alive] = rgamma(n = sum(!alive), 
                            shape = s + 1, 
                            rate = beta + tau[!alive]) 
      }
      
      mu[mu==0 | log(mu) < -30] <- exp(-30) # avoid numeric overflow
      return (mu)
    }
    
    # draw individual tau 
    draw_tau = function(lambda, mu, t.x, T.cal){
      
      N  = length(t.x)
      
      mu_lam = mu + lambda
      t_diff <- T.cal - t.x
      
      # sample alive
      p_alive = 1 / (1+(mu/mu_lam)*(exp(mu_lam*t_diff)-1))
      alive = p_alive > runif(n=N)
      
      # sample tau
      tau = numeric(N)
      
      # Case: still alive - left truncated exponential distribution -> [T.cal, Inf]
      if (any(alive)) {
        tau[alive]  = T.cal[alive] + rexp(sum(alive), mu[alive])
      }
      
      # Case: churned     - double truncated exponential distribution -> [t.x, T.cal]
      if (any(!alive)) {
        mu_lam_t.x   = pmin(700, mu_lam[!alive] * t.x[!alive])
        mu_lam_T.cal = pmin(700, mu_lam[!alive] * T.cal[!alive])
        
        rand        = runif(n=sum(!alive))
        tau[!alive] = -log( (1-rand)*exp(-mu_lam_t.x) + rand*exp(-mu_lam_T.cal)) / mu_lam[!alive]
      }
      
      return(tau)
      
    }
    
    
    # update the latent variables one by one; notice that updated param gets to be fed into the next step
    latent_variables['lambda', ] = draw_lambda(gamma_parameters['r'], gamma_parameters['alpha'], data$x, data$T.cal, latent_variables['tau', ])
    latent_variables['mu', ] = draw_mu(gamma_parameters['s'], gamma_parameters['beta'], latent_variables['tau', ], data$T.cal)
    latent_variables['tau', ] = draw_tau(latent_variables['lambda', ], latent_variables['mu', ], data$t.x, data$T.cal)
    latent_variables['alive', ] = as.numeric(latent_variables["tau",] > data$T.cal)
  
  return (latent_variables)
}


run_single_chain = function(chain_id=1, data, init_param, num_iter=3000, burn_in = 1000, thin = 50, trace=100){
  
  # dimension 
  num_cust = dim(data)[1]
  num_draws = (num_iter-1)%/%thin + 1
  
  # output 
  latent_variables_draws = array(NA_real_, dim = c(num_draws, 4, num_cust))
  dimnames(latent_variables_draws)[[2]] <- c("lambda", "mu", "tau", "alive")
  gamma_parameters_draws = array(NA_real_, dim = c(num_draws, 4))
  dimnames(gamma_parameters_draws)[[2]] <- c("r", "alpha", "s", "beta")
  
  
  # set initial params 
  hyper_prior = rep(1e-3, 8)
  
  # gamma_parameters
  gamma_parameters = gamma_parameters_draws[1, ]
  gamma_parameters['r'] = init_param$r
  gamma_parameters['alpha'] = init_param$alpha
  gamma_parameters['s'] = init_param$s
  gamma_parameters['beta'] = init_param$beta
  
  # latent_variables 
  latent_variables = latent_variables_draws[1,,]
  latent_variables['lambda', ] = mean(data$x) / mean(ifelse(data$t.x==0, data$T.cal, data$t.x))
  latent_variables['tau',] = data$t.x + 0.5/latent_variables["lambda",] 
  latent_variables['alive',] = as.numeric(latent_variables["tau",] > data$T.cal)
  latent_variables['mu', ] = 1/latent_variables["tau"]
  
  # Very good way to set up the initial params
  for (step in 1:(burn_in + num_iter)){
    if (step %% trace==0) cat("step:", step, "of chain", chain_id, "\n")
    
    if (step-burn_in > 0 & (step-1-burn_in)%%thin==0){
      iter <- (step-1-burn_in)%/%thin + 1
      latent_variables_draws[iter, , ] = latent_variables
      gamma_parameters_draws[iter, ] = gamma_parameters
    }
  
    latent_variables = draw_latent_variables(data, latent_variables, gamma_parameters)
    gamma_parameters = draw_gamma_parameters(latent_variables, gamma_parameters, hyper_prior)

  }
  
  output = list(latent_param = latent_variables_draws, gamma_param = gamma_parameters_draws)
  
  return (output)
}


cz_mcmc_multi_core = function(data, init_param=NULL, chains=2, ncores=2, num_iter=3000,  burn_in = 1000, thin = 50, trace=100){

  if (is.null(init_param)) {
    try({
      df = data[sample(nrow(data), min(nrow(data), 1000)),]
      init_param = BTYD::pnbd.EstimateParameters(df)
      names(init_param) = c("r", "alpha", "s", "beta")
      init_param = as.list(init_param)
    }, silent=TRUE)
    if (is.null(init_param)) init_param <- list(r=1, alpha=1, s=1, beta=1)
    cat("set param_init:", paste(round(unlist(init_param), 4), collapse=", "), "\n")
  }
  
  if (ncores>1) cat('Multiprocessing with', ncores, 'cores for', chains, 'chains \n')
  
  draws = lapply(1:chains, function(i) run_single_chain(i, data, init_param, num_iter, burn_in))
  
  return(list(latent_param = lapply(1:chains, function(i) draws[[i]]$latent_param), 
              gamma_param = lapply(1:chains, function(i) draws[[i]]$gamma_param)))
}



########################################## Helper function ##########################################################
calc.running.mean = function(vec){
  running.mean = rep(0, length(vec))
  for (i in 1:length(vec)){
    running.mean[i] = mean(vec[1:i])
  }
  return (running.mean)
}



