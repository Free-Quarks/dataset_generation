import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, N, I0, R0, t):
    S0 = N - I0 - R0
    S = [S0]
    I = [I0]
    R = [R0]
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        dS = -beta * S[-1] * I[-1] / N
        dI = beta * S[-1] * I[-1] / N - gamma * I[-1]
        dR = gamma * I[-1]
        S.append(S[-1] + dt * dS)
        I.append(I[-1] + dt * dI)
        R.append(R[-1] + dt * dR)
    return S, I, R

beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
t = np.linspace(0, 100, 1000)

S, I, R = SIR_model(beta, gamma, N, I0, R0, t)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model')
plt.grid(True)
plt.show()
