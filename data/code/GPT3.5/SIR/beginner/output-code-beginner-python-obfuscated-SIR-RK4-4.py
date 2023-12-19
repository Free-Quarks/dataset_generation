import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, S0, I0, R0, T):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = np.linspace(0, T, T+1)
    dt = t[1] - t[0]
    for i in range(T):
        dS_dt = -beta * S[i] * I[i] / N
        dI_dt = beta * S[i] * I[i] / N - gamma * I[i]
        dR_dt = gamma * I[i]
        S.append(S[i] + dt * dS_dt)
        I.append(I[i] + dt * dI_dt)
        R.append(R[i] + dt * dR_dt)
    return S, I, R

beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
T = 100

S, I, R = SIR_model(beta, gamma, S0, I0, R0, T)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

