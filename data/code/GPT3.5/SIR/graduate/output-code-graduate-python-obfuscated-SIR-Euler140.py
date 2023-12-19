import numpy as np
import matplotlib.pyplot as plt

def sir_model(S0, I0, R0, beta, gamma, t):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR = gamma * I[i-1]
        S.append(S[i-1] + dS * dt)
        I.append(I[i-1] + dI * dt)
        R.append(R[i-1] + dR * dt)
    return S, I, R

S0 = 999
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
t = np.linspace(0, 100, 1000)
S, I, R = sir_model(S0, I0, R0, beta, gamma, t)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
