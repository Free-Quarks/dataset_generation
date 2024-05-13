import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(N, beta, sigma, alpha, gamma, mu, delta, rho, t_max):
    # Define initial conditions
    S0 = N-1
    I0 = 1
    D0 = 0
    A0 = 0
    R0 = 0
    T0 = 0
    E0 = 0
    H0 = 0
    
    # Define arrays to store the values
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    D = np.zeros(t_max)
    A = np.zeros(t_max)
    R = np.zeros(t_max)
    T = np.zeros(t_max)
    E = np.zeros(t_max)
    H = np.zeros(t_max)
    
    # Assign initial values
    S[0] = S0
    I[0] = I0
    D[0] = D0
    A[0] = A0
    R[0] = R0
    T[0] = T0
    E[0] = E0
    H[0] = H0
    
    # Euler's method to solve the differential equations
    for t in range(1, t_max):
        S[t] = S[t-1] - beta*S[t-1]*I[t-1]/N - rho*S[t-1]*A[t-1]/N
        E[t] = E[t-1] + beta*S[t-1]*I[t-1]/N + rho*S[t-1]*A[t-1]/N - sigma*E[t-1]
        I[t] = I[t-1] + sigma*E[t-1] - alpha*I[t-1] - gamma*I[t-1] - mu*I[t-1]
        A[t] = A[t-1] + alpha*I[t-1] - delta*A[t-1]
        H[t] = H[t-1] + gamma*I[t-1] - mu*H[t-1]
        D[t] = D[t-1] + delta*A[t-1] + mu*(I[t-1] + H[t-1])
        R[t] = R[t-1] + delta*A[t-1] + mu*I[t-1]
        T[t] = T[t-1] + mu*H[t-1]
    
    return S, E, I, A, H, D, R, T


N = 100000
beta = 0.3
sigma = 0.02
alpha = 0.2
gamma = 0.05
mu = 0.01
delta = 0.01
rho = 0.05
t_max = 100

S, E, I, A, H, D, R, T = sidarthe_model(N, beta, sigma, alpha, gamma, mu, delta, rho, t_max)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(A, label='Asymptomatic')
plt.plot(H, label='Hospitalized')
plt.plot(D, label='Deceased')
plt.plot(R, label='Recovered')
plt.plot(T, label='Tested')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.legend()
plt.show()
