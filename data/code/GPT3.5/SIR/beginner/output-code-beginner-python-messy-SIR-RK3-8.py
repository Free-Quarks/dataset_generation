import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(beta, gamma, S0, I0, R0, t0, tmax, N, dt):
    # Initialize arrays
    t = np.arange(t0, tmax, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Runge-Kutta 3rd order method
    for i in range(1, len(t)):
        k1 = dt * (-beta * S[i-1] * I[i-1] / N)
        k2 = dt * (-beta * (S[i-1] + 0.5 * k1) * (I[i-1] + 0.5 * k1) / N)
        k3 = dt * (-beta * (S[i-1] - k1 + 2 * k2) * (I[i-1] - k1 + 2 * k2) / N)
        S[i] = S[i-1] + (1/6) * (k1 + 4 * k2 + k3)
        
        k1 = dt * (beta * S[i-1] * I[i-1] / N - gamma * I[i-1])
        k2 = dt * (beta * (S[i-1] + 0.5 * k1) * (I[i-1] + 0.5 * k1) / N - gamma * (I[i-1] + 0.5 * k1))
        k3 = dt * (beta * (S[i-1] - k1 + 2 * k2) * (I[i-1] - k1 + 2 * k2) / N - gamma * (I[i-1] - k1 + 2 * k2))
        I[i] = I[i-1] + (1/6) * (k1 + 4 * k2 + k3)
        
        R[i] = R[i-1] + dt * gamma * I[i-1]
        
    return t, S, I, R


# Parameters
beta = 0.25
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
N = 1000
T = 100
dt = 0.1

# Run simulation
t, S, I, R = SIR_RK3(beta, gamma, S0, I0, R0, 0, T, N, dt)

# Plot results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR Model using RK3')
plt.legend()
plt.show()
