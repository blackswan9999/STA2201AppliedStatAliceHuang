
data {
  int<lower=0> N; //number of obs
  vector[N] maize; //transformed maize
  vector[N] nitrogen; // nitrogen
  vector[N] water; // water
  int<lower=0> J; //number of soil types
  int<lower = 0, upper = J> soil[N]; //group membership
}


parameters {
  vector[J] alpha;
  vector[J] beta;
  vector[J] gamma;
  real mu_alpha;
  real mu_beta;
  real mu_gamma;
  real<lower=0> sigma_y;
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;
  real<lower=0> sigma_gamma;
}

model{
  vector[N] y_hat;
  
  for(i in 1:N){
    y_hat[i] = alpha[soil[i]] + beta[soil[i]]*nitrogen[i] + gamma[soil[i]]*water[i];
  }  

  //priors
  mu_alpha ~ normal(0,1);
  mu_beta ~ normal(0,1);
  mu_gamma ~ normal(0,1);
  sigma_y ~ normal(0,1);
  sigma_alpha ~ normal(0,1);
  sigma_beta ~ normal(0,1);
  sigma_gamma ~ normal(0,1);
  
  //group level model
  alpha ~ normal(mu_alpha, sigma_alpha);
  beta ~ normal(mu_beta, sigma_beta);
  gamma ~ normal(mu_gamma, sigma_gamma);
  
    //likelihood
  maize ~ normal(y_hat, sigma_y);
}

generated quantities{
  vector[N] y_hat;
  
  for(i in 1:N){
    y_hat[i] = alpha[soil[i]] + beta[soil[i]]*nitrogen[i] + gamma[soil[i]]*water[i];
  }  
}
