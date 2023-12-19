import numpy as np
import matplotlib.pyplot as plt

# Function to implement the SIR model using RK2

def sir_rk2_model(beta, gamma, S0, I0, R0, N, t_max, dt):
    t = np.arange(0, t_max + dt, dt)
    n = len(t)
    S = np.zeros(n)
    I = np.zeros(n)
    R = np.zeros(n)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(1, n):
        k1 = dt * (beta * S[i - 1] * I[i - 1] / N)
        k2 = dt * (beta * (S[i - 1] - k1 / 2) * (I[i - 1] - k1 / 2) / N)
        S[i] = S[i - 1] - k2
        k1 = dt * (beta * S[i - 1] * I[i - 1] / N - gamma * I[i - 1])
        k2 = dt * (beta * (S[i - 1] - k1 / 2) * (I[i - 1] - k1 / 2) / N - gamma * (I[i - 1] - k1 / 2))
        I[i] = I[i - 1] - k2
        R[i] = N - S[i] - I[i]
    return t, S, I, R

# Example usage

beta = 0.2
# Infection rate

gamma = 0.1
# Recovery rate

S0 = 990
# Initial susceptible population

I0 = 10
# Initial infected population

R0 = 0
# Initial recovered population

N = 1000
# Total population

t_max = 100
# Simulation time

dt = 0.1
# Time step

t, S, I, R = sir_rk2_model(beta, gamma, S0, I0, R0, N, t_max, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model - RK2')
plt.show()
