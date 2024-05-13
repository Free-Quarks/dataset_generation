import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, sigma, gamma, initial_conditions, t_start, t_end, dt):
    S0, E0, I0, R0 = initial_conditions
    N = S0 + I0 + R0
    t = np.arange(t_start, t_end, dt)
    S = np.zeros_like(t)
    E = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dE = beta * S[i-1] * I[i-1] / N - sigma * E[i-1]
        dI = sigma * E[i-1] - gamma * I[i-1]
        dR = gamma * I[i-1]
        S[i] = S[i-1] + dt * dS
        E[i] = E[i-1] + dt * dE
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR
    return S, E, I, R

# Example usage
t_start = 0
t_end = 100
dt = 0.1
beta = 0.5
sigma = 0.1
gamma = 0.05
initial_conditions = [999, 1, 0, 0]  # S0, E0, I0, R0

S, E, I, R = seir_model(beta, sigma, gamma, initial_conditions, t_start, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()
