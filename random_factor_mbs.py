import numpy as np

def random_factor_mbs(lambda_user):
    x_LOS_list = np.zeros(lambda_user)
    x_NLOS_list = np.zeros(lambda_user)
    z_list = np.zeros(lambda_user)
    
    m = 1
    v_LOS = 10**(4/10)  # Creating log-normal shadowing for LOS with unit mean and variance of 4 dB
    v_NLOS = 10**(6/10)  # Creating log-normal shadowing for NLOS with unit mean and variance of 6 dB

    mu_LOS = np.log((m**2) / np.sqrt(v_LOS + m**2))
    sigma_LOS = np.sqrt(np.log(v_LOS / (m**2) + 1))
    
    mu_NLOS = np.log((m**2) / np.sqrt(v_NLOS + m**2))
    sigma_NLOS = np.sqrt(np.log(v_NLOS / (m**2) + 1))
    
    for i in range(lambda_user):
        x_LOS_list[i] = np.random.lognormal(mu_LOS, sigma_LOS)
        x_NLOS_list[i] = np.random.lognormal(mu_NLOS, sigma_NLOS)
        z_list[i] = np.random.exponential(1)  # Creating fast fading gain coefficients with exponential distribution (unit mean)
    
    return x_LOS_list, x_NLOS_list, z_list