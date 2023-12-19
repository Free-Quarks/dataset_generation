import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(beta, gamma, S0, I0, R0, t_max, dt):
    N = S0 + I0 + R0
    t = np.arange(0, t_max, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0

    def dSdt(S, I, beta):
        return -beta * S * I / N

    def dIdt(S, I, beta, gamma):
        return beta * S * I / N - gamma * I

    def dRdt(I, gamma):
        return gamma * I

    for i in range(1, len(t)):
        k1_s = dSdt(S[i-1], I[i-1], beta)
        k1_i = dIdt(S[i-1], I[i-1], beta, gamma)
        k1_r = dRdt(I[i-1], gamma)

        k2_s = dSdt(S[i-1] + 0.5 * dt * k1_s, I[i-1] + 0.5 * dt * k1_i, beta)
        k2_i = dIdt(S[i-1] + 0.5 * dt * k1_s, I[i-1] + 0.5 * dt * k1_i, beta, gamma)
        k2_r = dRdt(I[i-1] + 0.5 * dt * k1_i, gamma)

        k3_s = dSdt(S[i-1] + 0.5 * dt * k2_s, I[i-1] + 0.5 * dt * k2_i, beta)
        k3_i = dIdt(S[i-1] + 0.5 * dt * k2_s, I[i-1] + 0.5 * dt * k2_i, beta, gamma)
        k3_r = dRdt(I[i-1] + 0.5 * dt * k2_i, gamma)

        k4_s = dSdt(S[i-1] + dt * k3_s, I[i-1] + dt * k3_i, beta)
        k4_i = dIdt(S[i-1] + dt * k3_s, I[i-1] + dt * k3_i, beta, gamma)
        k4_r = dRdt(I[i-1] + dt * k3_i, gamma)

        S[i] = S[i-1] + (dt / 6.0) * (k1_s + 2 * k2_s + 2 * k3_s + k4_s)
        I[i] = I[i-1] + (dt / 6.0) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
        R[i] = R[i-1] + (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)

    return t, S, I, R


beta = 0.3
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
t_max = 100
dt = 0.1

t, S, I, R = SIR_RK4(beta, gamma, S0, I0, R0, t_max, dt)

plt.figure(figsize=(10, 6))
plt.plot(t, S, label='S', color='blue')
plt.plot(t, I, label='I', color='red')
plt.plot(t, R, label='R', color='green')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK4')
plt.legend()
plt.grid(True)
plt.show()
