import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, S, I, R, t):
    N = S + I + R
    dt = t[1] - t[0]
    S_new = np.zeros_like(S)
    I_new = np.zeros_like(I)
    R_new = np.zeros_like(R)
    S_new[0] = S[0]
    I_new[0] = I[0]
    R_new[0] = R[0]

    for i in range(1, len(t)):
        k1_S = -beta * S[i-1] * I[i-1] / N
        k1_I = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k1_R = gamma * I[i-1]

        S_half = S[i-1] + 0.5 * dt * k1_S
        I_half = I[i-1] + 0.5 * dt * k1_I
        R_half = R[i-1] + 0.5 * dt * k1_R

        k2_S = -beta * S_half * I_half / N
        k2_I = beta * S_half * I_half / N - gamma * I_half
        k2_R = gamma * I_half

        S_new[i] = S[i-1] + dt * (2 * k2_S - k1_S) / 6
        I_new[i] = I[i-1] + dt * (2 * k2_I - k1_I) / 6
        R_new[i] = R[i-1] + dt * (2 * k2_R - k1_R) / 6

    return S_new, I_new, R_new


# Example usage
beta = 0.3
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
t = np.linspace(0, 100, 1000)

S, I, R = SIR_model(beta, gamma, S0, I0, R0, t)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
