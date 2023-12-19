import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(beta, gamma, S0, I0, R0, t_max, dt):
    N = S0 + I0 + R0
    t = np.arange(0, t_max, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)

    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        k1 = -beta * S[i-1] * I[i-1] / N
        l1 = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k2 = -beta * (S[i-1] + dt * k1/2) * (I[i-1] + dt * l1/2) / N
        l2 = beta * (S[i-1] + dt * k1/2) * (I[i-1] + dt * l1/2) / N - gamma * (I[i-1] + dt * l1/2)

        S[i] = S[i-1] + dt * k2
        I[i] = I[i-1] + dt * l2
        R[i] = R[i-1] + gamma * (I[i-1] + dt * l1/2)

    return S, I, R


# Example usage
beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0

t_max = 100
dt = 0.1

S, I, R = SIR_RK2(beta, gamma, S0, I0, R0, t_max, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK2')
plt.legend()
plt.show()
