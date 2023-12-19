import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, N, I0, R0, T):
    S0 = N - I0 - R0
    dsdt = np.zeros(T+1)
    drdt = np.zeros(T+1)
    didt = np.zeros(T+1)
    t = np.linspace(0, T, T+1)
    dsdt[0] = -beta * S0 * I0 / N
    drdt[0] = gamma * I0
    didt[0] = beta * S0 * I0 / N - gamma * I0

    for i in range(T):
        dsdt[i+1] = -beta * S0 * didt[i] / N
        drdt[i+1] = gamma * didt[i]
        didt[i+1] = beta * S0 * didt[i] / N - gamma * didt[i]
        S0 = S0 + (dsdt[i] + dsdt[i+1]) / 2
        R0 = R0 + (drdt[i] + drdt[i+1]) / 2
        I0 = I0 + (didt[i] + didt[i+1]) / 2

    return t, dsdt, drdt, didt

# Example usage
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
T = 100

t, dsdt, drdt, didt = SIR_model(beta, gamma, N, I0, R0, T)

plt.plot(t, dsdt, label='Susceptible')
plt.plot(t, drdt, label='Recovered')
plt.plot(t, didt, label='Infected')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
