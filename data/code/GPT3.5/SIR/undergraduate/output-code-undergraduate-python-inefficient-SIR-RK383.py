import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, S0, I0, R0, t_total):
    dt = 0.1
    N = S0 + I0 + R0
    t = np.arange(0, t_total, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))

    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR = gamma * I[i-1]

        k1_S = dt * dS
        k1_I = dt * dI
        k1_R = dt * dR

        k2_S = dt * (-beta * (S[i-1] + 0.5 * k1_S) * (I[i-1] + 0.5 * k1_I) / N)
        k2_I = dt * (beta * (S[i-1] + 0.5 * k1_S) * (I[i-1] + 0.5 * k1_I) / N - gamma * (I[i-1] + 0.5 * k1_I))
        k2_R = dt * gamma * (I[i-1] + 0.5 * k1_I)

        k3_S = dt * (-beta * (S[i-1] - k1_S + 2 * k2_S) * (I[i-1] - k1_I + 2 * k2_I) / N)
        k3_I = dt * (beta * (S[i-1] - k1_S + 2 * k2_S) * (I[i-1] - k1_I + 2 * k2_I) / N - gamma * (I[i-1] - k1_I + 2 * k2_I))
        k3_R = dt * gamma * (I[i-1] - k1_I + 2 * k2_I)

        S[i] = S[i-1] + (1/6) * (k1_S + 4 * k2_S + k3_S)
        I[i] = I[i-1] + (1/6) * (k1_I + 4 * k2_I + k3_I)
        R[i] = R[i-1] + (1/6) * (k1_R + 4 * k2_R + k3_R)

    return S, I, R


beta = 0.2
gamma = 0.1
S0 = 1000
I0 = 1
R0 = 0
t_total = 100

S, I, R = sir_model(beta, gamma, S0, I0, R0, t_total)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
