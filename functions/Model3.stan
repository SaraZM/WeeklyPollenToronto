data {
  int<lower=1> n; // number of sites
  int<lower=1> t; // number of weeks
  int<lower=1> nobs;// total number of observed values
  int<lower=1> nmis; // total number of missing values
  real east[n]; // coordinates in the east direction
  real north[n]; // coordinates in the north direction
  vector<lower=0>[nobs] y; // vector of observed values
  vector[n] zerosn; // vector of zeros of dimension n
  int<lower=1> vectimeobs[nobs]; //vector of observed weeks at each position in the vector
  int<lower=1> vecsiteobs[nobs]; //vector of observed sites at each position in the vector
  int<lower=0> vectimemis[nmis]; //vector of missing weeks at each position in the vector
  int<lower=0> vecsitemis[nmis]; //vector of missing sites at each position in the vector
  int<lower=0> ncovs; // number of covariates
  vector[ncovs] covs[n]; // matrix of covariates
  vector[3] t_covs[t]; // matrix of climatic factors
  vector[t] tc_brightness[n]; // TC brightness
  vector[t] tc_greenness[n]; // TC greenness
}


parameters{
  real theta0; // initial value state vector
  vector [t] theta; // state vector
  vector[3] alpha; // temporal covariates coeff
  real<lower=0> w_sd; // variance state vector
  real<lower=0> w2_sd; // variance probability
  real<lower=0> tau; // variance
  vector<lower=0>[nmis] y_miss; // vector of missing values
  vector[2] betas; // coordinates coeff
  vector[ncovs] lur; // land-use coeff
  vector[t] alpha_t; // climatic coeff
  real alpha0; // initial value for prob
  vector[2] tc; // TC coeff
  }


transformed parameters{

  vector[nobs] mu; // vector of the mean for the observed values
  vector[nmis] mu_mis; //vector of the mean values for missing values
  vector[t] logit_rho = alpha_t;
  vector<lower=0, upper = 1>[t] rho = inv_logit(logit_rho);

  real tau_sq = pow(tau,2);

  // computing the mean of the process for the observed values
  for(i in 1:nobs){
        mu[i] = theta[vectimeobs[i]] + 
                 betas[1]*east[vecsiteobs[i]] + betas[2]*north[vecsiteobs[i]]+ 
               lur'*covs[vecsiteobs[i]] + alpha'*t_covs[vectimeobs[i]]+ 
              tc[1]*tc_brightness[vecsiteobs[i], vectimeobs[i]] + 
    tc[2]*tc_greenness[vecsiteobs[i],vectimeobs[i]];
  }
  
  // computing the mean of the process for the missing values
  for(i in 1:nmis){
    mu_mis[i] = theta[vectimemis[i]] + 
                betas[1]*east[vecsitemis[i]] + betas[2]*north[vecsitemis[i]]+ 
               lur'*covs[vecsitemis[i]] + alpha'*t_covs[vectimemis[i]] + 
              tc[1]*tc_brightness[vecsitemis[i], vectimemis[i]]+ 
    tc[2]*tc_greenness[ vecsitemis[i],vectimemis[i]];

  }
  
  
} // end of transformed parameters

model{

  //state equations
  theta0 ~ normal(0,10);
  theta[1] ~ normal(theta0,w_sd);
  
  alpha0 ~ normal(0,10);
  alpha_t[1] ~ normal(alpha0, w2_sd);

  for(j in 2:t){
    theta[j] ~ normal(theta[j-1],w_sd);
    alpha_t[j] ~ normal(alpha_t[j-1],w2_sd);
  }
 

  betas ~ normal(0,10);
  tau ~ cauchy(0,1);
  w_sd ~ normal(0,1);
  w2_sd ~ normal(0,1);



  for(i in 1:nobs){
    if(y[i] == 0)
      1 ~ bernoulli(rho[vectimeobs[i]]);
      else{
        0 ~ bernoulli( rho[vectimeobs[i]]);
        y[i] ~ lognormal(mu[i],tau);
      }
    
  }
  
  //model for the missing values
  for(i in 1:nmis){
    if(y_miss[i] == 0)
      1 ~ bernoulli(rho[vectimemis[i]]);
      else{
        0 ~ bernoulli(rho[vectimemis[i]]);
         y_miss[i] ~ lognormal(mu_mis[i],tau);
      }
  }
  
} 

generated quantities {
  vector[nobs] log_lik;
  vector[nobs] y_fit;
  int<lower=0> y_pi;
  vector[nobs] mu_fit;
  
  for (i in 1:nobs) {
    if(y[i] == 0) {
      log_lik[i] = bernoulli_logit_lpmf(1 | logit_rho[vectimeobs[i]]);
    }else{
      log_lik[i] = lognormal_lpdf(y[i] | mu[i], tau);
    }      
  }
  
  for(i in 1:nobs){
    y_fit[i] = 0;
    y_pi = bernoulli_rng(rho[vectimeobs[i]]);
    if(y_pi == 0){
        y_fit[i] = lognormal_rng(mu[i],tau);
    
  }
  }

}// end of model
