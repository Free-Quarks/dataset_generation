import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, t_max, I0, N):
    def dS_dt(S, I):
        return -beta * S * I / N

    def dI_dt(S, I):
        return beta * S * I / N - gamma * I

    def dR_dt(I):
        return gamma * I

    t = np.linspace(0, t_max, t_max + 1)
    S = np.zeros(t_max + 1)
    I = np.zeros(t_max + 1)
    R = np.zeros(t_max + 1)

    I[0] = I0
    S[0] = N - I0

    for i in range(t_max):
        dS = dS_dt(S[i], I[i])
        dI = dI_dt(S[i], I[i])
        dR = dR_dt(I[i])

        S[i+1] = S[i] + dS
        I[i+1] = I[i] + dI
        R[i+1] = R[i] + dR

    return S, I, R


beta = 0.2
gamma = 0.1
t_max = 100
I0 = 10
N = 1000

S, I, R = SIR_model(beta, gamma, t_max, I0, N)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
