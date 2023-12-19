import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(beta, gamma, N, I0, R0, t_max, dt):
    def dSdt(S, I):
        return -beta * S * I / N

    def dIdt(S, I):
        return beta * S * I / N - gamma * I

    def dRdt(I):
        return gamma * I

    t = np.arange(0, t_max, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = N - I0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        k1S = dSdt(S[i-1], I[i-1])
        k1I = dIdt(S[i-1], I[i-1])
        k1R = dRdt(I[i-1])

        k2S = dSdt(S[i-1] + 0.5*dt*k1S, I[i-1] + 0.5*dt*k1I)
        k2I = dIdt(S[i-1] + 0.5*dt*k1S, I[i-1] + 0.5*dt*k1I)
        k2R = dRdt(I[i-1] + 0.5*dt*k1I)

        k3S = dSdt(S[i-1] + 0.5*dt*k2S, I[i-1] + 0.5*dt*k2I)
        k3I = dIdt(S[i-1] + 0.5*dt*k2S, I[i-1] + 0.5*dt*k2I)
        k3R = dRdt(I[i-1] + 0.5*dt*k2I)

        k4S = dSdt(S[i-1] + dt*k3S, I[i-1] + dt*k3I)
        k4I = dIdt(S[i-1] + dt*k3S, I[i-1] + dt*k3I)
        k4R = dRdt(I[i-1] + dt*k3I)

        S[i] = S[i-1] + (dt/6) * (k1S + 2*k2S + 2*k3S + k4S)
        I[i] = I[i-1] + (dt/6) * (k1I + 2*k2I + 2*k3I + k4I)
        R[i] = R[i-1] + (dt/6) * (k1R + 2*k2R + 2*k3R + k4R)

    return S, I, R

beta = 0.5
gamma = 0.2
N = 1000
I0 = 1
R0 = 0
T = 100
dt = 0.1

S, I, R = SIR_RK4(beta, gamma, N, I0, R0, T, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SIR Model using RK4')
plt.legend()
plt.show()
