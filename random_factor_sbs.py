import numpy as np

def random_factor_sbs(lambda_user, num_sbs):
    x_list_sbs = np.zeros((num_sbs, lambda_user))
    z_list_sbs = np.zeros((num_sbs, lambda_user))
    R_list = np.zeros((num_sbs, lambda_user))
    
    var_dB = 5.8
    v = 10**(var_dB / 10)
    m = 1  # mean
    mu = np.log((m**2) / np.sqrt(v + m**2))
    sigma = np.sqrt(np.log(v / (m**2) + 1))
    
    for s in range(num_sbs):
        for j in range(lambda_user):
            x_list_sbs[s, j] = np.random.lognormal(mu, sigma)
            z_list_sbs[s, j] = np.random.exponential(1)
            
            v = 5
            R_list[s, j] = np.random.gamma(v, 1 / v)
    
    return x_list_sbs, z_list_sbs, R_list