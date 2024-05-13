import numpy as np
import matplotlib.pyplot as plt

# Function implementing the SIDARTHE model

def sidarthe_model(N, I0, R0, D0, T, beta, gamma, delta, alpha, theta, rho):
    # Initialize arrays to store the values
    S = np.zeros(T)
    I = np.zeros(T)
    D = np.zeros(T)
    R = np.zeros(T)
    A = np.zeros(T)
    T = np.zeros(T)
    H = np.zeros(T)
    E = np.zeros(T)
    N = np.zeros(T)

    # Set initial values
    I[0] = I0
    R[0] = R0
    D[0] = D0
    S[0] = N - I[0] - R[0] - D[0]
    A[0] = 0
    T[0] = 0
    H[0] = 0
    E[0] = 0

    # Euler method to solve the differential equations
    for t in range(1, T):
        dS = -beta * I[t-1] * S[t-1] / N[t-1]
        dI = beta * I[t-1] * S[t-1] / N[t-1] - (gamma + delta + alpha) * I[t-1]
        dD = delta * I[t-1] - (theta + rho) * D[t-1]
        dR = gamma * I[t-1] + theta * D[t-1]
        dA = alpha * I[t-1]
        dT = rho * D[t-1]
        dH = delta * I[t-1]
        dE = beta * I[t-1] * S[t-1] / N[t-1]

        S[t] = S[t-1] + dS
        I[t] = I[t-1] + dI
        D[t] = D[t-1] + dD
        R[t] = R[t-1] + dR
        A[t] = A[t-1] + dA
        T[t] = T[t-1] + dT
        H[t] = H[t-1] + dH
        E[t] = E[t-1] + dE

    return S, I, D, R, A, T, H, E

# Parameters
N = 1000000
I0 = 100
R0 = 0
D0 = 0
T = 100
beta = 0.2
gamma = 0.1
alpha = 0.01
delta = 0.01
theta = 0.01
rho = 0.01

# Run the model
data = sidarthe_model(N, I0, R0, D0, T, beta, gamma, delta, alpha, theta, rho)

# Plot the results
plt.plot(data[0], label='S')
plt.plot(data[1], label='I')
plt.plot(data[2], label='D')
plt.plot(data[3], label='R')
plt.plot(data[4], label='A')
plt.plot(data[5], label='T')
plt.plot(data[6], label='H')
plt.plot(data[7], label='E')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.show()
