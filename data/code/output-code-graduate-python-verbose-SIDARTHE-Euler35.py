import numpy as np
import matplotlib.pyplot as plt


# Function implementing the SIDARTHE model
def sidarthe_model(N, beta, rho, sigma, gamma1, gamma2, gamma3, delta, alpha, theta, t_max):
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    D = np.zeros(t_max)
    A = np.zeros(t_max)
    R = np.zeros(t_max)
    T = np.zeros(t_max)
    H = np.zeros(t_max)
    E = np.zeros(t_max)
    N = N - 1
    S[0] = N
    E[0] = 1
    I[0] = 0
    D[0] = 0
    A[0] = 0
    R[0] = 0
    T[0] = 0
    H[0] = 0

    # Euler's method to solve the differential equations
    for t in range(t_max - 1):
        dS = -beta * S[t] * (I[t] + rho * A[t]) / N
        dE = beta * S[t] * (I[t] + rho * A[t]) / N - sigma * E[t]
        dI = sigma * E[t] - gamma1 * I[t] - gamma2 * I[t] - gamma3 * I[t] - delta * I[t] - alpha * I[t]
        dD = gamma3 * I[t]
        dA = gamma1 * I[t]
        dR = gamma2 * I[t]
        dT = delta * I[t]
        dH = alpha * I[t]
        S[t + 1] = S[t] + dS
        E[t + 1] = E[t] + dE
        I[t + 1] = I[t] + dI
        D[t + 1] = D[t] + dD
        A[t + 1] = A[t] + dA
        R[t + 1] = R[t] + dR
        T[t + 1] = T[t] + dT
        H[t + 1] = H[t] + dH

    return S, E, I, D, A, R, T, H


# Example usage
N = 10000
beta = 0.2
rho = 0.5
sigma = 0.1
gamma1 = 0.05
gamma2 = 0.1
gamma3 = 0.01
delta = 0.005
alpha = 0.005
theta = 0.5
t_max = 100

S, E, I, D, A, R, T, H = sidarthe_model(N, beta, rho, sigma, gamma1, gamma2, gamma3, delta, alpha, theta, t_max)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(D, label='Deceased')
plt.plot(A, label='Asymptomatic')
plt.plot(R, label='Recovered')
plt.plot(T, label='Transferred')
plt.plot(H, label='Hospitalized')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model Simulation')
plt.show()
