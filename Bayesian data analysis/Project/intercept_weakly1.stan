data {
        int <lower=0> N;
        vector[N] y;  
        vector[N] x1; 
        vector[N] x2; 
        int zipID[N]; 
        int numzipID; 
        int <lower=0> N_test; 
        vector[N_test] x1_test; 
        vector[N_test] x2_test; 
        int zipID_test[N_test]; 
}
parameters {
        real alpha[numzipID];   
        real beta[2]; 
        real <lower=0> sigma;   
        real alpha_p;
        real <lower=0> alpha_sigma_p;
}    
model {
        // priors
        alpha ~ normal(alpha_p, alpha_sigma_p);
        beta[1] ~ normal(0.5, 1);
        beta[2] ~ normal(0.1, 1);
        sigma ~ cauchy(0, 2.5);
        // hyperpriors
        alpha_p ~ normal(10, 10);
        alpha_sigma_p ~ cauchy(0, 2.5);
        for (i in 1:N) {
            y[i] ~ normal(alpha[zipID[i]] + beta[1]*x1[i] + beta[2]*x2[i], sigma);
        }
}  
generated quantities {
        vector[N] log_lik;
        vector[N] y_reps;
        vector[N_test] y_preds;
        for (i in 1:N){
            log_lik[i] = normal_lpdf(y[i] | alpha[zipID[i]] + beta[1]*x1[i] + beta[2]*x2[i], sigma);
            y_reps[i] = normal_rng(alpha[zipID[i]] + beta[1]*x1[i] + beta[2]*x2[i], sigma);
        }
        for (i in 1:N_test){
            y_preds[i] = normal_rng(alpha[zipID_test[i]] + beta[1]*x1_test[i] + beta[2]*x2_test[i], sigma);
        }
}