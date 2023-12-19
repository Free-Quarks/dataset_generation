import numpy as np
import matplotlib.pyplot as plt


def sir_model(S0, I0, R0, beta, gamma, t_max):
    N = S0 + I0 + R0
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    t = np.linspace(0, t_max-1, t_max)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, t_max):
        dSdt = -beta * S[i-1] * I[i-1] / N
        dIdt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        S[i] = S[i-1] + dSdt
        I[i] = I[i-1] + dIdt
        R[i] = R[i-1] + dRdt
    
    return S, I, R


# Example usage
S0 = 999
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
t_max = 100

S, I, R = sir_model(S0, I0, R0, beta, gamma, t_max)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

