import sys
sys.path.append('/Users/yueli/src')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# data = np.loadtxt("../premier_league_2013_2014.dat", delimiter=',')
# data = data.astype(int)

# propose function / Q of MH procedure
def w(data, theta, eta, sigma, tau_1, alpha, beta):
    tau_0 = .0001
    log_p = 0
    for g in range(380):
        log_p += - np.exp(theta[0] + theta[1][0, data[g, 2]] - theta[1][1, data[g, 3]]) + \
                 (theta[0] + theta[1][0, data[g, 2]] - theta[1][1, data[g, 3]]) * data[g, 0] - \
                 np.exp(theta[1][0, data[g, 3]] - theta[1][1, data[g, 2]]) + \
                 (theta[1][0, data[g, 3]] - theta[1][1, data[g, 2]]) * data[g, 1]
    log_p += - tau_0/2 * theta[0]**2
    log_p += - eta[2]/2 * np.sum((theta[1][0,:] - eta[0]) * (theta[1][0, :] - eta[0])) \
             + np.log(eta[2]) * 10
    log_p += - eta[3]/2 * np.sum((theta[1][1,:] - eta[1]) * (theta[1][1, :] - eta[1])) \
             + np.log(eta[3]) * 10
    log_p += - tau_1/2 * eta[0]**2 - tau_1/2 * eta[1]**2 + \
                 (alpha - 1) * np.log(eta[2] * eta[3]) - 1/beta * (eta[2] + eta[3])
    # -log(q(theta))
    # log_p += 1 / (2 * sigma ** 2) * (theta[0] ** 2 + np.sum(theta[1] * theta[1]) + np.sum(eta * eta))
    return log_p

# one step of MH procedure
def mh_one_step(theta, eta, data, sigma, tau_1, alpha, beta):
    # run one step of M-H algorithm
    theta_propose = (sigma * np.random.randn(1) + theta[0], sigma * np.random.randn(2, 20)+theta[1])
    theta_propose[1][:, 0] = .0
    # how to propose eta? Looks like we can not directly do it.
    eta_propose = sigma * np.random.randn(4) + eta
    eta_propose[2] = max(eta_propose[2], 1e-6)
    eta_propose[3] = max(eta_propose[3], 1e-6)
    
    w_propose = w(data, theta_propose, eta_propose, sigma, tau_1, alpha, beta)
    # print(w_propose)
    w_current = w(data, theta, eta, sigma, tau_1, alpha, beta)
    a = w_propose - w_current
    # print(w_propose, w_current)
    if a >= 0:
        reject = 0
    else:
        reject = np.random.binomial(1, 1-np.exp(a))[0]
    if reject:
        theta_new, eta_new = theta, eta
    else:
        theta_new, eta_new = theta_propose, eta_propose
    return theta_new, eta_new, reject


def mh(data, sigma, burn_in, T, N):
    # How to initialize?
    tau_1 = .0001
    alpha = .1
    beta  = .1
    theta_initial = (sigma*np.random.randn(1), sigma*np.random.randn(2, 20))
    theta_initial[1][:, 0] = .0
    eta_initial = sigma*np.random.randn(4) + .1
    eta_initial[2] = max(eta_initial[2], 1e-6)
    eta_initial[3] = max(eta_initial[3], 1e-6)

    # burn_in steps:
    print('Burn in:')
    home_burnin = []
    theta = theta_initial
    eta = eta_initial
    for i in tqdm.tqdm(range(burn_in)):
        theta, eta, reject = mh_one_step(theta, eta, data, sigma, tau_1, alpha, beta)
        home_burnin.append(theta[0])

    # MCMC chain:
    print('MCMC chain:')
    theta_seq = []
    reject_seq = []
    for i in tqdm.tqdm(range(N)):
        for t in range(T):
            theta, eta, reject = mh_one_step(theta, eta, data, sigma, tau_1, alpha, beta)
        theta_seq.append(theta)
        reject_seq.append(reject)
        # print(i, theta[0], reject)

    return theta_seq, reject_seq, home_burnin


def mcmc_run(data, sigma, T, burn_in=5000, N=5000):
    print('sigma=', sigma, '; t=', T)
    file_name = 'sigma_' + str(sigma) + '_t_' + str(T)
    theta_seq, reject_seq, home_burnin = mh(data, sigma, burn_in, T, N)
    home_mc = []
    for theta in theta_seq:
        home_mc.append(theta[0])
    np.save('result/' + file_name + '_home_burnin.npy', home_burnin)
    np.save('result/' + file_name + '_home_mc.npy', home_mc)
    np.save('result/' + file_name + '_theta.npy', theta_seq)

    plt.plot(home_burnin)
    plt.title('Trace plot for burn-in period')
    plt.savefig('result/' + file_name + '_home_burnin_plot.png', format='png')
    plt.close()

    plt.plot(home_mc)
    plt.title('Trace plot for MCMC samples')
    plt.savefig('result/' + file_name + '_home_mc_plot.png', format='png')
    plt.close()
   
    print('Rejection rate:', np.sum(reject_seq))
    print('Finished.')


if __name__ == '__main__':
    data = np.loadtxt("../premier_league_2013_2014.dat", delimiter=',')
    data = data.astype(int)
    mcmc_run(data, .005, 1)
    mcmc_run(data, .05,  1)
    mcmc_run(data, .5,   1)
    mcmc_run(data, .005, 5)
    mcmc_run(data, .05,  5)
    mcmc_run(data, .5,   5)
    mcmc_run(data, .005, 20)
    mcmc_run(data, .05,  20)
    mcmc_run(data, .5,   20)
    mcmc_run(data, .005, 50)
    mcmc_run(data, .05,  50)
    mcmc_run(data, .5,   50)
