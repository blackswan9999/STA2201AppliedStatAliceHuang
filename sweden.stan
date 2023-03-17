data {
  int<lower=1> N;
  int<lower=0> deaths[N];
  int age[N];
  vector[N] pop;
}

parameters {
  real<lower=0> alpha;
  real<lower=0> beta;
}

model {
  //alpha baseline mortality
  // beta INCREASE age in year, what does mortality increase by
  for (n in 1:N) {
    target += poisson_log_lpmf(deaths[n] | log(alpha) + beta*age[n] + log(pop[n]));
  }
  // Priors
  target += normal_lpdf(alpha | 0.0015, 0.01);
  target += normal_lpdf(beta | 0.005, 0.01);
}

generated quantities {
  vector[N] log_lik;    // pointwise log-likelihood for LOO
  vector[N] log_deaths_rep; // replications from posterior predictive dist

  for (n in 1:N) {
  real rate_hat_n = log(alpha) + log(pop[n]) + beta * age[n];
  log_lik[n] = poisson_log_lpmf(deaths[n] | rate_hat_n);
  // To avoid erros like the below during the warmup. 
  // [2] "  Exception: poisson_log_rng: Log rate parameter is 21.2382, but must be less than 20.7944
  // Check posterior predictive. 
  if (rate_hat_n > 20.7944) {
    log_deaths_rep[n] = poisson_log_rng(20.7944);
  } else {
    log_deaths_rep[n] = poisson_log_rng(rate_hat_n);
  }
}
}