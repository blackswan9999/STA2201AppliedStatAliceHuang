data {
  int<lower=0> N; // number of age categories
  int<lower=0> M; // number of year categories
  int deaths[N,M];
  matrix[N,M] age;
  matrix[N,M] pop;
}

parameters {
  vector<lower=0>[M] alpha;
  vector<lower=0>[M] beta;
}

transformed parameters{
  matrix[N,M] mort;
  // loop through years
  for(m in 1:M){
      mort[:,m] = beta[m]*age[:,m] + log(alpha[m]*pop[:,m]);
    }
}

model {
  for(m in 1:M){
    target += normal_lpdf(alpha[m]| 0.0015, 0.01);
    target += normal_lpdf(beta[m]|0.005, 0.01);
  }
  
  for(n in 1:N){
    for(m in 1:M){
      target += poisson_log_lpmf(deaths[n,m]|mort[n,m]);
    }
  }
}
