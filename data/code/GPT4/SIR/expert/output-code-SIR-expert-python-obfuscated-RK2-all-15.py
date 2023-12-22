import matplotlib.pyplot as plt
import numpy as np
def SIR_model(t, Y, beta, gamma):
    S, I, R = Y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def rk2(Y, t, dt, model, beta, gamma):
    k1 = model(t, Y, beta, gamma)
    k2 = model(t + dt, [Y[i] + dt/2 * k1[i] for i in range(len(Y))], beta, gamma)
    return [Y[i] + dt * k2[i] for i in range(len(Y))]
S0, I0, R0 = 990, 10, 0
beta, gamma = 0.5, 0.1
T = 50
dt = 0.1
N = int(T / dt) + 1
t = np.linspace(0, T, N)
S = np.empty(N)
I = np.empty(N)
R = np.empty(N)
S[0], I[0], R[0] = S0, I0, R0
for i in range(N-1):
    S[i+1], I[i+1], R[i+1] = rk2([S[i], I[i], R[i]], t[i], dt, SIR_model, beta, gamma)
plt.figure()
plt.plot(t, S, label='S')
plt.plot(t, I, label='I')
plt.plot(t, R, label='R')
plt.legend()
plt.show()
