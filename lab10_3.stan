data {
  int<lower=0> N; // number of observations
  int<lower=0> T; //number of years
  vector[N] y; //log ratio
  vector[N] se; // standard error around observations
  vector[T] years; // unique years of study
  int<lower=0> year_i[N]; // year index of observations
  int<lower=0> P; // number of years to project
}

parameters {
  vector[T] mu; //expected value of time series
  real<lower=0> sigma; //variance of random walkd

}

model {
  
  y ~ normal(mu[year_i], se);
  
  mu[1] ~ normal(0, 1);
  mu[2] ~ normal(0, 1);
  mu[3:T] ~ normal(2*mu[2:(T-1)] - mu[1:(T-2)],sigma);
  sigma ~ normal(0,1);
}

generated quantities{
  vector[P] mu_p;
  mu_p[1] = normal_rng(mu[T-1], sigma);
  mu_p[2] = normal_rng(mu[T], sigma);
  for(i in 3:P){
    mu_p[i] = normal_rng(2*mu[(i-1)] - mu[(i-2)], sigma);
  }
}