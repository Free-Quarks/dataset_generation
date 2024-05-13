import numpy as np
import matplotlib.pyplot as plt


def SIDARTHE_RK2(S0, I0, D0, A0, R0, T0, H0, E0, N, beta, sigma, gamma, alpha, rho, theta, delta, t_end, dt):
    # Define the initial conditions
    S = np.zeros(t_end+1)
    I = np.zeros(t_end+1)
    D = np.zeros(t_end+1)
    A = np.zeros(t_end+1)
    R = np.zeros(t_end+1)
    T = np.zeros(t_end+1)
    H = np.zeros(t_end+1)
    E = np.zeros(t_end+1)
    
    S[0] = S0
    I[0] = I0
    D[0] = D0
    A[0] = A0
    R[0] = R0
    T[0] = T0
    H[0] = H0
    E[0] = E0
    
    # Define the time vector
    t = np.arange(0, t_end+1)
    
    # Implement the RK2 method
    for i in range(t_end):
        k1 = dt * SIDARTHE_derivatives(S[i], I[i], D[i], A[i], R[i], T[i], H[i], E[i], N, beta, sigma, gamma, alpha, rho, theta, delta)
        k2 = dt * SIDARTHE_derivatives(S[i] + 0.5 * k1[0], I[i] + 0.5 * k1[1], D[i] + 0.5 * k1[2], A[i] + 0.5 * k1[3], R[i] + 0.5 * k1[4], T[i] + 0.5 * k1[5], H[i] + 0.5 * k1[6], E[i] + 0.5 * k1[7], N, beta, sigma, gamma, alpha, rho, theta, delta)
        
        S[i+1] = S[i] + k2[0]
        I[i+1] = I[i] + k2[1]
        D[i+1] = D[i] + k2[2]
        A[i+1] = A[i] + k2[3]
        R[i+1] = R[i] + k2[4]
        T[i+1] = T[i] + k2[5]
        H[i+1] = H[i] + k2[6]
        E[i+1] = E[i] + k2[7]
    
    return S, I, D, A, R, T, H, E


def SIDARTHE_derivatives(S, I, D, A, R, T, H, E, N, beta, sigma, gamma, alpha, rho, theta, delta):
    dS = -beta * S * (I + D + A + R + T + H + E) / N
    dI = (1 - sigma) * beta * S * (I + D + A + R + T + H + E) / N - gamma * I
    dD = sigma * beta * S * (I + D + A + R + T + H + E) / N - alpha * D
    dA = theta * alpha * D - rho * A
    dR = (1 - theta) * alpha * D + (1 - delta) * gamma * I
    dT = delta * gamma * I
    dH = rho * A
    dE = sigma * beta * S * (I + D + A + R + T + H + E) / N
    
    return dS, dI, dD, dA, dR, dT, dH, dE


# Define the parameters
S0 = 1000000
I0 = 100
D0 = 20
A0 = 10
R0 = 0
T0 = 0
H0 = 0
E0 = 0
N = S0 + I0 + D0 + A0 + R0 + T0 + H0 + E0
beta = 0.35
sigma = 0.2
gamma = 0.2
alpha = 0.2
rho = 0.1
theta = 0.1
delta = 0.01
t_end = 100
dt = 1

# Run the model
S, I, D, A, R, T, H, E = SIDARTHE_RK2(S0, I0, D0, A0, R0, T0, H0, E0, N, beta, sigma, gamma, alpha, rho, theta, delta, t_end, dt)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(D, label='Deceased')
plt.plot(A, label='Asymptomatic')
plt.plot(R, label='Recovered')
plt.plot(T, label='Tested')
plt.plot(H, label='Hospitalized')
plt.plot(E, label='Exposed')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIDARTHE Model with RK2')
plt.legend()
plt.show()
