import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, S0, I0, R0, N, t_end, dt):
    def f(y):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return np.array([dSdt, dIdt, dRdt])

    t = np.arange(0, t_end, dt)

    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))

    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        k1 = dt * f([S[i-1], I[i-1], R[i-1]])
        k2 = dt * f([S[i-1] + k1[0]/2, I[i-1] + k1[1]/2, R[i-1] + k1[2]/2])
        S[i] = S[i-1] + k2[0]
        I[i] = I[i-1] + k2[1]
        R[i] = R[i-1] + k2[2]

    return S, I, R


# Example usage
beta = 0.2
gamma = 0.1
S0 = 99
I0 = 1
R0 = 0
N = 100

t_end = 100
dt = 0.1

S, I, R = SIR_model(beta, gamma, S0, I0, R0, N, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
