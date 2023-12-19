import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, N, I0, R0, tmax):
    dt = 0.01
    t = np.arange(0, tmax, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0
    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] * dt
        dI = (beta * S[i-1] * I[i-1] - gamma * I[i-1]) * dt
        dR = gamma * I[i-1] * dt
        S[i] = S[i-1] + dS
        I[i] = I[i-1] + dI
        R[i] = R[i-1] + dR
    return S, I, R


beta = 0.2
gamma = 0.1
N = 1000
I0 = 10
R0 = 0
tmax = 100

S, I, R = SIR_model(beta, gamma, N, I0, R0, tmax)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model')
plt.show()
