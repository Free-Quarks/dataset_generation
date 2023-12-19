import numpy as np
import matplotlib.pyplot as plt


def SIR_model(S, I, R, beta, gamma, t):
    N = S + I + R
    S_dot = -beta * S * I / N
    I_dot = beta * S * I / N - gamma * I
    R_dot = gamma * I
    return S_dot, I_dot, R_dot


def RK4(S, I, R, beta, gamma, t, dt):
    S_values = [S]
    I_values = [I]
    R_values = [R]

    while t > 0:
        S_dot, I_dot, R_dot = SIR_model(S, I, R, beta, gamma, t)

        k1_S = dt * S_dot
        k1_I = dt * I_dot
        k1_R = dt * R_dot

        k2_S = dt * SIR_model(S + k1_S/2, I + k1_I/2, R + k1_R/2, beta, gamma, t + dt/2)[0]
        k2_I = dt * SIR_model(S + k1_S/2, I + k1_I/2, R + k1_R/2, beta, gamma, t + dt/2)[1]
        k2_R = dt * SIR_model(S + k1_S/2, I + k1_I/2, R + k1_R/2, beta, gamma, t + dt/2)[2]

        k3_S = dt * SIR_model(S + k2_S/2, I + k2_I/2, R + k2_R/2, beta, gamma, t + dt/2)[0]
        k3_I = dt * SIR_model(S + k2_S/2, I + k2_I/2, R + k2_R/2, beta, gamma, t + dt/2)[1]
        k3_R = dt * SIR_model(S + k2_S/2, I + k2_I/2, R + k2_R/2, beta, gamma, t + dt/2)[2]

        k4_S = dt * SIR_model(S + k3_S, I + k3_I, R + k3_R, beta, gamma, t + dt)[0]
        k4_I = dt * SIR_model(S + k3_S, I + k3_I, R + k3_R, beta, gamma, t + dt)[1]
        k4_R = dt * SIR_model(S + k3_S, I + k3_I, R + k3_R, beta, gamma, t + dt)[2]

        S += (k1_S + 2 * k2_S + 2 * k3_S + k4_S) / 6
        I += (k1_I + 2 * k2_I + 2 * k3_I + k4_I) / 6
        R += (k1_R + 2 * k2_R + 2 * k3_R + k4_R) / 6

        S_values.append(S)
        I_values.append(I)
        R_values.append(R)

        t -= dt

    return S_values, I_values, R_values


S = 999
I = 1
R = 0
beta = 0.3
gamma = 0.1
t = 100
dt = 0.1

S_values, I_values, R_values = RK4(S, I, R, beta, gamma, t, dt)

plt.plot(S_values, label='Susceptible')
plt.plot(I_values, label='Infected')
plt.plot(R_values, label='Recovered')

plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
