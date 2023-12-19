import numpy as np
import matplotlib.pyplot as plt


def SIR_RK4(beta, gamma, S0, I0, R0, t_max, dt):
    N = S0 + I0 + R0
    t = np.linspace(0, t_max, int(t_max/dt) + 1)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0

    def f_S(t, S, I):
        return -beta * S * I / N

    def f_I(t, S, I):
        return beta * S * I / N - gamma * I

    def f_R(t, I):
        return gamma * I

    for i in range(len(t)-1):
        k1_S = f_S(t[i], S[i], I[i])
        k1_I = f_I(t[i], S[i], I[i])
        k1_R = f_R(t[i], I[i])

        k2_S = f_S(t[i] + dt/2, S[i] + dt/2 * k1_S, I[i] + dt/2 * k1_I)
        k2_I = f_I(t[i] + dt/2, S[i] + dt/2 * k1_S, I[i] + dt/2 * k1_I)
        k2_R = f_R(t[i] + dt/2, I[i] + dt/2 * k1_I)

        k3_S = f_S(t[i] + dt/2, S[i] + dt/2 * k2_S, I[i] + dt/2 * k2_I)
        k3_I = f_I(t[i] + dt/2, S[i] + dt/2 * k2_S, I[i] + dt/2 * k2_I)
        k3_R = f_R(t[i] + dt/2, I[i] + dt/2 * k2_I)

        k4_S = f_S(t[i] + dt, S[i] + dt * k3_S, I[i] + dt * k3_I)
        k4_I = f_I(t[i] + dt, S[i] + dt * k3_S, I[i] + dt * k3_I)
        k4_R = f_R(t[i] + dt, I[i] + dt * k3_I)

        S[i+1] = S[i] + dt/6 * (k1_S + 2*k2_S + 2*k3_S + k4_S)
        I[i+1] = I[i] + dt/6 * (k1_I + 2*k2_I + 2*k3_I + k4_I)
        R[i+1] = R[i] + dt/6 * (k1_R + 2*k2_R + 2*k3_R + k4_R)

    return S, I, R


beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
t_max = 100
dt = 0.1

S, I, R = SIR_RK4(beta, gamma, S0, I0, R0, t_max, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK4')
plt.show()
