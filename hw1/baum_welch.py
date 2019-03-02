import numpy as np
from scipy.stats import multivariate_normal


def e_step(X, pi, A, mu, sigma2):
    """E-step: forward-backward message passing"""
    # Messages and sufficient statistics
    N, T, K = X.shape
    M = A.shape[0]
    alpha = np.zeros([N,T,M])  # [N,T,M]
    alpha_sum = np.zeros([N,T])  # [N,T], normalizer for alpha
    beta = np.zeros([N,T,M])  # [N,T,M]
    gamma = np.zeros([N,T,M])  # [N,T,M]
    xi = np.zeros([N,T-1,M,M])  # [N,T-1,M,M]

    # Forward messages
    for n in range(N):
        for t in range(T):
            for i in range(M):
                if t == 0:
                    alpha[n, 0, i] = pi[i] * (2 * np.pi * sigma2[i])**(-K/2) * np.exp(
                        - np.sum((X[n, 0, :] - mu[i, :]) * (X[n, 0, :] - mu[i, :]))/(2*sigma2[i]))
                else:
                    alpha[n, t, i] = np.sum(alpha[n, t-1, :] * A[:, i]) * (2 * np.pi * sigma2[i])**(-K/2) * np.exp(- np.sum((X[n, t, :] - mu[i, :]
                        ) * (X[n, t, :] - mu[i, :]))/(2*sigma2[i]))
            alpha_sum[n, t] = np.sum(alpha[n, t, :])
            alpha[n, t, :] = alpha[n, t, :] / alpha_sum[n, t]

    # Backward messages
    for n in range(N):
        for t in range(T-1, -1, -1):
            for i in range(M):
                if t == T-1:
                    beta[n, t, i] = 1
                else:
                    beta[n, t, i]  = np.sum([beta[n, t+1, j] * A[i, j] * (2 * np.pi * sigma2[j])**(-K/2) 
                                             * np.exp(- np.sum((X[n, t+1, :] - mu[j, :]) * (X[n, t+1, :] - mu[j, :]))
                                                      /(2*sigma2[j])) for j in range(M)]) / alpha_sum[n, t+1] 

    # Sufficient statistics
    gamma = alpha * beta
    for n in range(N):
        for t in range(T):
            gamma[n, t, :] = gamma[n, t, :] / np.sum(gamma[n, t, :])

    # only xi remained!
    for n in range(N):
        for t in range(T-1):
            for i in range(M):
                for j in range(M):
                    xi[n, t, i, j] = alpha[n, t, i] * A[i, j] * (2 * np.pi * sigma2[j])**(-K/2) * np.exp(
                        - np.sum((X[n, t+1, :] - mu[j, :]) * (X[n, t+1, :] - mu[j, :]))/(
                            2*sigma2[j])) * beta[n, t+1, j] 
            xi[n, t, :, :] = xi[n, t, :, :] / np.sum(xi[n, t, :, :])
    # Although some of them will not be used in the M-step, please still
    # return everything as they will be used in test cases
    return alpha, alpha_sum, beta, gamma, xi


def m_step(X, gamma, xi):
    """M-step: MLE"""
    # input: X, [N, T, K] gamma, [N, T, M] xi, [N,T-1,M,M]
    # initialization
    
    N, T, K = X.shape
    M = gamma.shape[2]
    pi = np.zeros(M)
    A = np.zeros([M, M])
    mu = np.zeros([M, K])
    sigma2 = np.zeros(M)

    # updating
    pi = np.sum(gamma[:,0,:], axis=0) / N
    pi = pi/np.sum(pi)
    xi_sum = np.sum(np.sum(xi, axis=0), axis=0)
    gamma_sum = np.sum(np.sum(gamma[:, 0:(T-1), :], axis=0), axis=0) #[M,]
    A = xi_sum / gamma_sum[:, None]
    gamma_sum = np.sum(np.sum(gamma, axis=0), axis=0) #[M,]
    for m in range(M):
        for k in range(K):
            mu[m, k] = np.sum(gamma[:, :, m] * X[:, :, k]) / gamma_sum[m]
    for m in range(M):
        X_norm = np.zeros([N, T])
        for n in range(N):
            for t in range(T):
                X_norm[n, t] = np.sum((X[n,t,:]-mu[m, :]) * (X[n,t,:]-mu[m, :]))
        sigma2[m] = np.sum(X_norm * gamma[:,:,m]) / (K * gamma_sum[m])
    
    return pi, A, mu, sigma2


def hmm_train(X, pi, A, mu, sigma2, em_step=20):
    """Run Baum-Welch algorithm."""
    for step in range(em_step):
        _, alpha_sum, _, gamma, xi = e_step(X, pi, A, mu, sigma2)
        pi, A, mu, sigma2 = m_step(X, gamma, xi)
        print(f"step: {step}  ln p(x): {np.einsum('nt->', np.log(alpha_sum))}")
    return pi, A, mu, sigma2


def hmm_generate_samples(N, T, pi, A, mu, sigma2):
    """Given pi, A, mu, sigma2, generate [N,T] samples."""
    M, K = mu.shape
    Y = np.zeros([N,T], dtype=int) 
    X = np.zeros([N,T,K], dtype=float)
    for n in range(N):
        Y[n,0] = np.random.choice(M, p=pi)  # [1,]
        X[n,0,:] = multivariate_normal.rvs(mu[Y[n,0],:], sigma2[Y[n,0]] * np.eye(K))  # [K,]
    for t in range(T - 1):
        for n in range(N):
            Y[n,t+1] = np.random.choice(M, p=A[Y[n,t],:])  # [1,]
            X[n,t+1,:] = multivariate_normal.rvs(mu[Y[n,t+1],:], sigma2[Y[n,t+1]] * np.eye(K))  # [K,]
    return X


def main():
    """Run Baum-Welch on a simulated toy problem."""
    # Generate a toy problem
    np.random.seed(12345)  # for reproducibility
    N, T, M, K = 10, 100, 4, 2
    pi = np.array([.0, .0, .0, 1.])  # [M,]
    A = np.array([[.7, .1, .1, .1],
                  [.1, .7, .1, .1],
                  [.1, .1, .7, .1],
                  [.1, .1, .1, .7]])  # [M,M]
    mu = np.array([[2., 2.],
                   [-2., 2.],
                   [-2., -2.],
                   [2., -2.]])  # [M,K]
    sigma2 = np.array([.2, .4, .6, .8])  # [M,]
    X = hmm_generate_samples(N, T, pi, A, mu, sigma2)

    # Run on the toy problem
    pi_init = np.random.rand(M)
    pi_init = pi_init / pi_init.sum()
    A_init = np.random.rand(M, M)
    A_init = A_init / A_init.sum(axis=-1, keepdims=True)
    mu_init = 2 * np.random.rand(M, K) - 1
    sigma2_init = np.ones(M)
    pi, A, mu, sigma2 = hmm_train(X, pi_init, A_init, mu_init, sigma2_init, em_step=20)
    print(pi)
    print(A)
    print(mu)
    print(sigma2)


if __name__ == '__main__':
    main()

