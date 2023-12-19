import numpy as np
import matplotlib.pyplot as plt

def simulate_SIR(beta, gamma, N, I0, R0, T):
    S0 = N - I0 - R0
    dt = 0.1
    t = np.linspace(0, T, int(T/dt)+1)
    S = np.empty(t.shape)
    I = np.empty(t.shape)
    R = np.empty(t.shape)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, t.size):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = (beta * S[i-1] * I[i-1] / N) - gamma * I[i-1]
        dR = gamma * I[i-1]

        S[i] = S[i-1] + dt * dS
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR

    return t, S, I, R

beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
T = 50

t, S, I, R = simulate_SIR(beta, gamma, N, I0, R0, T)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()

