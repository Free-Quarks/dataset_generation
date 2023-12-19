import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, init_conditions, t_max, dt):
    S0, I0, R0 = init_conditions
    N = S0 + I0 + R0
    t = np.arange(0, t_max, dt)

    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)

    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR = gamma * I[i-1]

        S_hat = S[i-1] + dt * dS
        I_hat = I[i-1] + dt * dI
        R_hat = R[i-1] + dt * dR

        S[i] = S[i-1] + (dt / 6) * (dS + 4 * dS_hat + dS_hat2)
        I[i] = I[i-1] + (dt / 6) * (dI + 4 * dI_hat + dI_hat2)
        R[i] = R[i-1] + (dt / 6) * (dR + 4 * dR_hat + dR_hat2)

    return S, I, R


# Example usage
beta = 0.2
gamma = 0.1
init_conditions = (990, 10, 0)
t_max = 100
dt = 0.1

S, I, R = sir_model(beta, gamma, init_conditions, t_max, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

