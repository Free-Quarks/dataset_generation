import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(N, beta, gamma, delta, alpha, rho, theta, sigma, t_max):
    # Initialize arrays
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    D = np.zeros(t_max)
    A = np.zeros(t_max)
    R = np.zeros(t_max)
    T = np.zeros(t_max)
    H = np.zeros(t_max)
    E = np.zeros(t_max)
    # Set initial conditions
    S[0] = N - 1
    I[0] = 1
    D[0] = 0
    A[0] = 0
    R[0] = 0
    T[0] = 0
    H[0] = 0
    E[0] = 0
    # Euler method
    dt = 0.1
    for t in range(1, t_max):
        S[t] = S[t-1] - (beta*S[t-1]*I[t-1]/N)*dt
        I[t] = I[t-1] + ((beta*S[t-1]*I[t-1]/N) - (gamma*I[t-1]) - (delta*I[t-1]))*dt
        D[t] = D[t-1] + (delta*I[t-1])*dt
        A[t] = A[t-1] + ((1-alpha)*gamma*I[t-1] - (rho*A[t-1]))*dt
        R[t] = R[t-1] + (rho*A[t-1])*dt
        T[t] = T[t-1] + (alpha*gamma*I[t-1])*dt
        H[t] = H[t-1] + (theta*alpha*gamma*I[t-1])*dt
        E[t] = E[t-1] + (sigma*theta*alpha*gamma*I[t-1])*dt
    # Return arrays
    return S, I, D, A, R, T, H, E


# Example usage
N = 1000
beta = 0.2
gamma = 0.1
alpha = 0.2
delta = 0.05
rho = 0.1
theta = 0.01
sigma = 0.01
t_max = 100

S, I, D, A, R, T, H, E = sidarthe_model(N, beta, gamma, delta, alpha, rho, theta, sigma, t_max)

plt.plot(S, label='S')
plt.plot(I, label='I')
plt.plot(D, label='D')
plt.plot(A, label='A')
plt.plot(R, label='R')
plt.plot(T, label='T')
plt.plot(H, label='H')
plt.plot(E, label='E')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.show()
