import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model_euler(S0, I0, R0, beta, gamma, T, dt):
    N = int(T/dt) + 1
    S, I, R = np.zeros(N), np.zeros(N), np.zeros(N)
    S[0], I[0], R[0] = S0, I0, R0
    for n in range(N-1):
        S[n+1] = S[n] - dt * beta * S[n] * I[n]
        I[n+1] = I[n] + dt * (beta * S[n] * I[n] - gamma * I[n])
        R[n+1] = R[n] + dt * gamma * I[n]
    return S, I, R

S, I, R = sir_model_euler(S0=0.99, I0=0.01, R0=0.0, beta=0.5, gamma=0.1, T=60, dt=0.1)

plt.figure(figsize=(10,6))
plt.plot(S, label="Susceptibles")
plt.plot(I, label="Infected")
plt.plot(R, label="Recovered")
plt.legend()
plt.title("SIR Model using Euler's method")
plt.show()
